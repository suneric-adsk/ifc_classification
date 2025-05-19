import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import transforms
from pathlib import Path
from .plyreader import read_ply

class PLYDataset(Dataset):

    def __init__(self, data_root, class_names, split="train", transform=None):
        self.transform = transform
        self.data_root = data_root
        self.class_names = class_names
        self.split = split
        self.file_paths = sorted(data_root.glob(f"**/{split}/*.ply"))

        self.cache = {}

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            pointcloud, label = self.cache[idx]
        else:
            path = self.file_paths[idx]
            df = read_ply(path)
            pointcloud = df["points"].to_numpy()
            class_name = path.parts[-3]
            label = self.class_names.index(class_name)
            self.cache[idx] = (pointcloud, label)

        if self.transform:
            pointcloud = self.transform(pointcloud)

        return pointcloud, label
    

class TranslatePointCloud:
    def __call__(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3]) # rotation
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3]) # translation
        translated_pc = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pc
        
class ShufflePointCloud:
    def __call__(self, pointcloud):
        copied_pc = pointcloud.copy()
        np.random.shuffle(copied_pc)
        return copied_pc
        
    
def get_dgcnn_dataloader(data_root, class_names, batch_size, eval_on_test=False):
    transform = transforms.Compose([
        TranslatePointCloud(),
        ShufflePointCloud()
    ])

    if eval_on_test:
        train_dataset = PLYDataset(data_root, class_names, split="train", transform=transform)
        val_dataset = PLYDataset(data_root, class_names, split="test")
    else:
        train_dataset = PLYDataset(data_root, class_names, split="train", transform=transform)
        val_dataset = PLYDataset(data_root, class_names, split="train")

        np.random.seed(42)
        perm = np.random.permutation(range(len(train_dataset)))
        train_len = int(0.7*len(train_dataset))
        train_dataset = Subset(train_dataset, perm[:train_len])
        val_dataset = Subset(val_dataset, perm[train_len:])
    
    print(f"Train Size: {len(train_dataset)}")
    print(f"Val Size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=len(train_dataset)%batch_size==1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    return train_loader, val_loader