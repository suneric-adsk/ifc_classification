import torch
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np


class MeshDataset(Dataset):

    def __init__(self, data_root, max_face, class_names, split='train'):
        self.data_root = data_root
        self.max_face = max_face
        self.split = split 
        self.file_paths = sorted(data_root.glob(f"**/{split}/*.npz"))
        self.class_names = class_names

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        class_name = file_path.parts[-3]
        label = self.class_names.index(class_name)
        
        data = np.load(file_path)
        face = data['faces']
        neighbor_idx = data['neighbors']

        n_point = len(face)
        if n_point < self.max_face:
            fill_face = []
            fill_neighbor_idx = []
            for i in range(self.max_face-n_point):
                idx = np.random.randint(0, n_point)
                fill_face.append(face[idx])
                fill_neighbor_idx.append(neighbor_idx[idx])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_idx = np.concatenate((neighbor_idx, np.array(fill_neighbor_idx)))

        face = torch.from_numpy(face)
        neighbor_idx = torch.from_numpy(neighbor_idx)
        target = torch.tensor(label, dtype=torch.long)
        data = torch.cat([face, neighbor_idx], dim=1)

        return data, target

    def __len__(self):
        return len(self.file_paths)


def get_meshnet_dataloader(data_root, class_names, batch_size, eval_on_test=False):
    if eval_on_test:
        train_dataset = MeshDataset(data_root, 2048, class_names, split="train")
        val_dataset = MeshDataset(data_root, 2048, class_names, split="test")
    else:
        train_dataset = MeshDataset(data_root, 2048, class_names, split="train")
        val_dataset = MeshDataset(data_root, 2048, class_names, split="train")
        np.random.seed(42)
        perm = np.random.permutation(range(len(train_dataset)))
        train_len = int(0.7*len(train_dataset))
        train_dataset = Subset(train_dataset, perm[:train_len])
        val_dataset = Subset(val_dataset, perm[train_len:])
        
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    return train_loader, val_loader