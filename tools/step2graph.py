import argparse
import pathlib
import os

import dgl
import numpy as np
import torch
from occwl.graph import face_adjacency
from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm
from multiprocessing.pool import Pool
from itertools import repeat
import signal

def bounding_box(input):
    pts = input[...,:3].reshape((-1,3))
    mask = input[...,6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces,:]
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    return np.array([[x.min(), y.min(), z.min()],[x.max(), y.max(), z.max()]])

def center_and_scale_uvgrid(input, bbox=None):
    # bounding box
    if bbox is None:
        bbox = bounding_box(input)
    # center and scale
    scale = 2.0 / np.linalg.norm(bbox[1] - bbox[0])
    center = 0.5 * (bbox[0] + bbox[1])
    input[..., :3] -= center
    input[..., :3] *= scale
    return input, center, scale

def build_graph(solid, crv_sample, srf_sample_u, srf_sample_v, center_and_scale=False):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    graph_face_feat = []
    for idx in graph.nodes:
        face = graph.nodes[idx]["face"]
        points = uvgrid(face, method="point", num_u=srf_sample_u, num_v=srf_sample_v)    
        normals = uvgrid(face, method="normal", num_u=srf_sample_u, num_v=srf_sample_v)
        visibility = uvgrid(face, method="visibility_status", num_u=srf_sample_u, num_v=srf_sample_v)
        masks = np.logical_or(visibility == 0, visibility == 1)
        face_feat = np.concatenate([points, normals, masks], axis=-1)
        graph_face_feat.append(face_feat)  
    graph_face_feat = np.asarray(graph_face_feat)

    graph_edge_feat = []
    for idx in graph.edges:
        edge = graph.edges[idx]["edge"]
        if not edge.has_curve():
            continue
        points = ugrid(edge, method="point", num_u=crv_sample)
        tangents = ugrid(edge, method="tangent", num_u=crv_sample)
        edge_feat = np.concatenate([points, tangents], axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    bbox = bounding_box(graph_face_feat)
    if (center_and_scale):
        graph_face_feat, center, scale = center_and_scale_uvgrid(graph_face_feat, bbox)
        graph_edge_feat[...,:3] -= center
        graph_edge_feat[...,:3] *= scale

    # convert face0-adj graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    # print("Build graph {} node, {} edges".format(self.dgl_graph.num_nodes(), self.dgl_graph.num_edges()))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph

def process_one_file(arguments):
    files, args = arguments
    step_file, bin_file = files
    try:
        solids = load_step(step_file) # Assume there's one solid per file
    except:
        print("translation failed")
        return 
    
    if len(solids) == 0:
        print("no solid found in the step file")
        return 
    
    graph_list, node_count = [], []
    for i in range(len(solids)):
        solid = solids[0]
        solid.set_transform_to_identity()
        graph = build_graph(solid, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples, center_and_scale=True)
        graph_list.append(graph)
        node_count.append(graph.num_nodes())

    counts_with_indices = [(count, i) for i, count in enumerate(node_count)]
    count_sorted = sorted(counts_with_indices, key=lambda x: x[0], reverse=True)

    # combined the most complex shapes
    combined_count = 0
    max_count = 5120 # for the limit of GPU memory
    combined_graph = []
    for count, idx in count_sorted:
        if combined_count + count < max_count:
            combined_count += count
            combined_graph.append(graph_list[idx])
        else:
            if combined_count > 0:
                break
    
    if len(combined_graph) == 0:
        print("no graph found")
        return 
    
    print("combined graph:", len(combined_graph), "total nodes:", combined_count)
    graph = dgl.merge(combined_graph)
    dgl.data.utils.save_graphs(bin_file, [graph])

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process(args):
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    parifiles = []
    for root, dirs, files in os.walk(input_path):
        for dir_name in dirs:
            dest_dir_path = os.path.join(output_path, os.path.relpath(os.path.join(root, dir_name), input_path))
            os.makedirs(dest_dir_path, exist_ok=True)
        
        for filename in files:
            src_file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(src_file_path, input_path)
            base_name, _ = os.path.splitext(relative_path)
            new_filename = base_name + ".bin"
            dest_file_path = os.path.join(output_path, new_filename)
            if not os.path.exists(dest_file_path):
                parifiles.append((src_file_path, dest_file_path))

    print("total files:",len(parifiles))
    pool = Pool(processes=args.num_processes, initializer=initializer)
    try:
        results = list(tqdm(pool.imap(process_one_file, zip(parifiles, repeat(args))), total=len(parifiles)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    print(f"Processed {len(results)} files.")


def main():
    parser = argparse.ArgumentParser(
        "Convert solid models to face-adjacency graphs with UV-grid features"
    )
    parser.add_argument("input", type=str, help="Input folder of STEP files")
    parser.add_argument("output", type=str, help="Output folder of DGL graph BIN files")
    parser.add_argument(
        "--curv_u_samples", type=int, default=15, help="Number of samples on each curve"
    )
    parser.add_argument(
        "--surf_u_samples",
        type=int,
        default=15,
        help="Number of samples on each surface along the u-direction",
    )
    parser.add_argument(
        "--surf_v_samples",
        type=int,
        default=15,
        help="Number of samples on each surface along the v-direction",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to use",
    )
    args = parser.parse_args()

    print(args.output)
    process(args)


if __name__ == "__main__":
    main()
