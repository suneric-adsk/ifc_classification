import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import transforms
from PIL import Image

def _get_transform(mode):
    if mode == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

class SingleViewDataset(Dataset):

    def __init__(self, root_dir, class_names, split="train", mode="train"):
        self.class_names = class_names
        self.root_dir = root_dir
        self.split = split
        self.file_paths = sorted(root_dir.glob(f"**/{split}/*.png"))
        self.transform = _get_transform(mode)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        class_name = path.parts[-3]
        label = self.class_names.index(class_name)

        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

class MultiViewDataset(Dataset):

    def __init__(self, root_dir, class_names, n_view, split="train", mode="train"):
        self.class_names = class_names
        self.n_view = n_view
        self.root_dir = root_dir
        self.file_paths = sorted(root_dir.glob(f"**/{split}/*.png"))
        self.file_paths = np.array(self.file_paths).reshape(-1, self.n_view)
        self.transform = _get_transform(mode)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        paths = self.file_paths[idx]
        class_name = paths[0].parts[-3]
        label = self.class_names.index(class_name)

        images = []
        for path in paths:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        return torch.stack(images), label
    

def get_svcnn_dataloader(data_root, class_names, batch_size, eval_on_test=False,mode="train"):
    
    if eval_on_test:
        train_dataset = SingleViewDataset(data_root, class_names, split="train", mode=mode)
        val_dataset = SingleViewDataset(data_root, class_names, split="test", mode=mode)
    else:
        train_dataset = SingleViewDataset(data_root, class_names, split="train", mode=mode)
        val_dataset = SingleViewDataset(data_root, class_names, split="train", mode=mode)
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

def get_mvcnn_dataloader(data_root, class_names, batch_size, n_view=12, eval_on_test=False, mode="train"):

    if eval_on_test:
        train_dataset = MultiViewDataset(data_root, class_names, n_view, split="train", mode=mode)
        val_dataset = MultiViewDataset(data_root, class_names, n_view, split="test", mode=mode)
    else:
        train_dataset = MultiViewDataset(data_root, class_names, n_view, split="train", mode=mode)
        val_dataset = MultiViewDataset(data_root, class_names, n_view, split="train", mode=mode)
        np.random.seed(42)
        perm = np.random.permutation(range(len(train_dataset)))
        train_len = int(0.7*len(train_dataset))
        train_dataset = Subset(train_dataset, perm[:train_len])
        val_dataset = Subset(val_dataset, perm[train_len:])
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    return train_loader, val_loader