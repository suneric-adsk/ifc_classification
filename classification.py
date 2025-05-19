import time 
import pathlib
import argparse
import json
import torch.nn as nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from models.utils.ply_dataset import *
from models.ifcnet_dgcnn import IFCNet_DGCNN

from models.utils.img_dataset import *
from models.modules.mvcnn import SVCNN, MVCNN
from models.ifcnet_mvcnn import IFCNet_MVCNN

from models.utils.mesh_dataset import *
from models.ifcnet_meshnet import IFCNet_MeshNet

from models.utils.stp_dataset import *
from models.ifcnet_uvnet import IFCNet_UVNet

def train_test_dgcnn(checkpoint_cb, class_names, logger, eval_on_test=False):
    with open("./trained/dgcnn_model/DGCNNParams.json", "r") as f:
        config = json.load(f)

    print("DGCNN Training")
    
    
    model = IFCNet_DGCNN(config) 
    data_path = pathlib.Path("./dataset/IFCNetCorePly/IFCNetCore")
    train_loader, val_loader = get_dgcnn_dataloader(
        data_root=data_path.absolute(),
        class_names=class_names,
        batch_size=int(config["batch_size"]),
        eval_on_test=eval_on_test
    )
    trainer.fit(model, train_loader, val_loader)

######

parser = argparse.ArgumentParser("IFC Classification")
parser.add_argument("traintest", choices=("train", "test"))
parser.add_argument("--type", type=str, default="view") # ply, mesh
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--epochs", type=int, default=100)

args = parser.parse_args()

results_path = pathlib.Path(f"./results/{args.type}")
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

month_day = time.strftime("%m%d")
hour_min_sec = time.strftime("%H%M%S")

checkpoint_cb = ModelCheckpoint(
    monitor="valid_balanced_accuracy_score",
    dirpath = str(results_path.joinpath(month_day, hour_min_sec)),
    filename="best",
    mode="max",
    save_top_k=10,
    save_last=True
)

logger = TensorBoardLogger(str(results_path), name=month_day, version=hour_min_sec)

with open("IFCNetCore_Classes.json", "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

trainer = Trainer(
    default_root_dir=str(results_path), 
    max_epochs=args.epochs,
    callbacks=[checkpoint_cb],
    logger=logger,
    accelerator='gpu',
    devices=1,
    gradient_clip_val=1.0
)

if args.type == "ply":
    with open("./trained/dgcnn_model/DGCNNParams.json", "r") as f:
        config = json.load(f)

    print("DGCNN training and test")
    model = IFCNet_DGCNN() 
    data_path = pathlib.Path("./dataset/IFCNetCorePly/IFCNetCore")
    if args.traintest == "train":
        train_loader, val_loader = get_dgcnn_dataloader(
            data_root=data_path.absolute(),
            class_names=class_names,
            batch_size=int(config["batch_size"]),
            eval_on_test=False
        )
        trainer.fit(model, train_loader, val_loader)
    else:
        # assert args.ckpt is not None, "No checkpoint provided"
        # model = IFCNet_DGCNN.load_from_checkpoint(args.ckpt)
        model = IFCNet_DGCNN()
        model.load_for_test(args.ckpt)
        _, test_loader = get_dgcnn_dataloader(
            data_root=data_path.absolute(),
            class_names=class_names,
            batch_size=int(config["batch_size"]),
            eval_on_test=True
        )
        trainer.test(model, dataloaders=[test_loader], ckpt_path=args.ckpt, verbose=False)

elif args.type == "view":
    with open("./trained/mvcnn_model/MVCNNParams.json", "r") as f:
        config = json.load(f)

    print("MVCNN training and test")
    data_path = pathlib.Path("./dataset/IFCNetCorePng/IFCNetCore")
    if args.traintest == "train":
        # 1st stage
        svcnn = SVCNN(n_class=len(class_names))
        svmodel = IFCNet_MVCNN()
        svmodel.set_model(model=svcnn, n_view=1) 
 
        sv_train_loader, sv_val_loader = get_svcnn_dataloader(
            data_root=data_path.absolute(),
            class_names=class_names,
            batch_size=int(config["batch_size"]),
            eval_on_test=False,
            mode="train"
        )

        sv_trainer = Trainer(
            default_root_dir=str(results_path), 
            max_epochs=30,
            callbacks=[checkpoint_cb],
            logger=logger,
            accelerator='gpu',
            devices=1,
            gradient_clip_val=1.0
        )
        sv_trainer.fit(svmodel, sv_train_loader, sv_val_loader)

        # 2nd stage
        n_view = int(config["num_views"])
        mvcnn = MVCNN(svmodel.model, n_view=n_view)
        mvmodel = IFCNet_MVCNN()
        mvmodel.set_model(model=svmodel.model, n_view=12)
        del svmodel

        mv_train_loader, mv_val_loader = get_mvcnn_dataloader(
            data_root=data_path.absolute(),
            class_names=class_names,
            batch_size=int(config["batch_size"]/n_view),
            n_view=n_view,
            eval_on_test=False,
            mode="train"
        )
        trainer.fit(mvmodel, mv_train_loader, mv_val_loader)
    else:
        model = IFCNet_MVCNN.load_from_checkpoint(args.ckpt)
        _, test_loader = get_mvcnn_dataloader(
            data_root=data_path.absolute(),
            class_names=class_names,
            batch_size=int(config["batch_size"]/config["num_views"]),
            n_view=int(config["num_views"]),
            eval_on_test=False,
            mode="test"
        )
        trainer.test(model, dataloaders=[test_loader], ckpt_path=args.ckpt, verbose=False)

elif args.type == "mesh":
    with open("./trained/meshnet_model/MeshNetParams.json", "r") as f:
        config = json.load(f)

    print("MeshNet training and test")
    model = IFCNet_MeshNet() 
    data_path = pathlib.Path("./dataset/IFCNetCoreNpz/IFCNetCore")
    if args.traintest == "train":
        train_loader, val_loader = get_meshnet_dataloader(
            data_root=data_path.absolute(),
            class_names=class_names,
            batch_size=int(config["batch_size"]),
            eval_on_test=False
        )
        trainer.fit(model, train_loader, val_loader)
    else:
        assert args.ckpt is not None, "No checkpoint provided"
        model = IFCNet_MeshNet.load_from_checkpoint(args.ckpt)
        _, test_loader = get_meshnet_dataloader(
            data_root=data_path.absolute(),
            class_names=class_names,
            batch_size=int(config["batch_size"]),
            eval_on_test=True
        )
        trainer.test(model, dataloaders=[test_loader], ckpt_path=args.ckpt, verbose=False)

elif args.type == "stp":
    print("UVNet training and test")
    data_path = pathlib.Path("./dataset/IFCNetCoreStp/IFCNetCore")
    if args.traintest == "train":
        uvnet = IFCNet_UVNet()
        train_loader, val_loader = get_uvnet_dataloader(
            data_root=data_path.absolute(),
            class_names=class_names,
            batch_size=32,
            eval_on_test=False,
            mode="train"
        )
        trainer.fit(uvnet, train_loader, val_loader)