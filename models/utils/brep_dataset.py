import os 
import sys
sys.path.append('..')
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
from occwl.io import load_step
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.shell import Shell
import dgl
from dgl.data.utils import load_graphs
from scipy.spatial.transform import Rotation
from torch import FloatTensor

def get_random_rotation():
    """Get a random rotation in 90 degree increments along the canonical axes"""
    axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    angles = [0.0, 90.0, 180.0, 270.0]
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)


def rotate_uvgrid(inp, rotation):
    """Rotate the node features in the graph by a given rotation"""
    Rmat = torch.tensor(rotation.as_matrix()).float()
    orig_size = inp[..., :3].size()
    inp[..., :3] = torch.mm(inp[..., :3].view(-1, 3), Rmat).view(
        orig_size
    )  # Points
    inp[..., 3:6] = torch.mm(inp[..., 3:6].view(-1, 3), Rmat).view(
        orig_size
    )  # Normals/tangents
    return inp

"""
Brep dataset for test only
"""
class BrepDataSet(Dataset):
    def __init__(self, root_dir, class_names, split="train", mode="train"):
        self.class_names = class_names
        self.root_dir = root_dir
        self.split = split
        self.file_paths = sorted(root_dir.glob(f"**/{split}/*.bin"))
        self.rotation = get_random_rotation() if mode=="train" else None


    def load_files(self, root_dir):
        """Load files in root_dir"""
        files = os.listdir(root_dir)
        return [f for f in files if f.endswith(".bin")]
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        class_name = path.parts[-3]
        label = self.class_names.index(class_name)
        graph = load_graphs(str(path))[0][0]
        graph.ndata["x"] = graph.ndata["x"].type(FloatTensor)
        graph.edata["x"] = graph.edata["x"].type(FloatTensor)
        if self.rotation is not None:
            graph.ndata["x"] = rotate_uvgrid(graph.ndata["x"], self.rotation)
            graph.edata["x"] = rotate_uvgrid(graph.edata["x"], self.rotation)
        return graph, label
    
def _collate(batch):
    batch_graph = dgl.batch([data[0] for data in batch])
    batch_label = torch.as_tensor([data[1] for data in batch])
    return batch_graph, batch_label

def get_uvnet_dataloader(data_root, class_names, batch_size, eval_on_test=False, mode="train"):
    
    if eval_on_test:
        train_dataset = BrepDataSet(data_root, class_names, split="train", mode=mode)
        val_dataset = BrepDataSet(data_root, class_names, split="test", mode=mode)
    else:
        train_dataset = BrepDataSet(data_root, class_names, split="train", mode=mode)
        val_dataset = BrepDataSet(data_root, class_names, split="train", mode=mode)
        np.random.seed(42)
        perm = np.random.permutation(range(len(train_dataset)))
        train_len = int(0.7*len(train_dataset))
        train_dataset = Subset(train_dataset, perm[:train_len])
        val_dataset = Subset(val_dataset, perm[train_len:])
        
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=_collate, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=_collate, num_workers=8)
    return train_loader, val_loader