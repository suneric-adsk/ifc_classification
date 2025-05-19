import numpy as np
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid
import dgl
import torch
from torch import FloatTensor

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
    dgl_graph.ndata["x"] = dgl_graph.ndata["x"].type(FloatTensor)
    dgl_graph.edata["x"] = dgl_graph.edata["x"].type(FloatTensor)
    return dgl_graph

    