import numpy as np
import torch
import lightning as L
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from .modules.uvnet import BrepUVNet
from .utils.plot_results import plot_confusion_matrix
import pathlib

def _uvnet_loss(pred, label):
    loss = F.cross_entropy(pred, label, reduction="mean")
    return loss 

def _permute_graph(graph):
    graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
    graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
    return graph


class IFCNet_UVNet(L.LightningModule):
    """
    PyTorch Lightning module to train/test the classifier.
    """

    def __init__(self):
        """
        Args:
            num_classes (int): Number of per-solid classes in the dataset
        """
        super().__init__()

        self.classnames = ["IfcAirTerminal", "IfcBeam", "IfcCableCarrierFitting", "IfcCableCarrierSegment", "IfcDoor", 
                           "IfcDuctFitting", "IfcDuctSegment", "IfcFurniture", "IfcLamp", "IfcOutlet", 
                           "IfcPipeFitting", "IfcPipeSegment", "IfcPlate", "IfcRailing", "IfcSanitaryTerminal", 
                           "IfcSlab", "IfcSpaceHeater", "IfcStair", "IfcValve", "IfcWall"]
        
        self.model = BrepUVNet(
            d_embed=512, dropout=0.3, num_classes=len(self.classnames)
        )

        self.learning_rate = 1e-4
        self.weight_decay = 2e-5
        self.loss_fn = _uvnet_loss
        self.dataload_cb = _permute_graph

        self.train_probs = []
        self.train_labels = []
        self.valid_probs = []
        self.valid_labels = []

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.dataload_cb:
            x = self.dataload_cb(x)
        y_hat, _ = self(x)
        loss = self.loss_fn(pred=y_hat, label=y)
        
        probs = F.softmax(y_hat, dim=1)
        self.train_probs.append(probs.cpu().detach().numpy())
        self.train_labels.append(y.cpu().numpy())

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.dataload_cb:
            x = self.dataload_cb(x)
        y_hat, _ = self(x)
        loss = self.loss_fn(pred=y_hat, label=y)

        probs = F.softmax(y_hat, dim=1)
        self.valid_probs.append(probs.cpu().detach().numpy())
        self.valid_labels.append(y.cpu().numpy())

        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        return loss   

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.dataload_cb:
            x = self.dataload_cb(x)
        y_hat, _ = self(x)

        probs = F.softmax(y_hat, dim=1)
        self.valid_probs.append(probs.cpu().detach().numpy())
        self.valid_labels.append(y.cpu().numpy())
    
    def on_train_epoch_end(self):
        probs = np.concatenate(self.train_probs)
        labels = np.concatenate(self.train_labels)
        metrics = self._calc_metrics(probs, labels, tag="train")
        self.train_labels = []
        self.train_probs = []
        self.log_dict(metrics)

    def on_validation_epoch_end(self):
        probs = np.concatenate(self.valid_probs)
        labels = np.concatenate(self.valid_labels)
        metrics = self._calc_metrics(probs, labels, tag="valid")
        self.valid_labels = []
        self.valid_probs = []
        self.log_dict(metrics)

    def on_test_epoch_end(self):
        probs = np.concatenate(self.valid_probs)
        labels = np.concatenate(self.valid_labels)
        metrics = self._calc_metrics(probs, labels, tag="test")
        self.valid_labels = []
        self.valid_probs = []
        self.log_dict(metrics)

        preds = np.argmax(probs, axis=1)
        confusion_mat = confusion_matrix(labels, preds)
        plot_confusion_matrix(confusion_mat, self.classnames, fname=pathlib.Path("./report/UVNet_confusion_matrix.png"))

    def _calc_metrics(self, probs, labels, tag):
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds, average="weighted")
        f1 = f1_score(labels, preds, average="weighted")

        return {
            f"{tag}_accuracy_score": acc,
            f"{tag}_balanced_accuracy_score": balanced_acc,
            f"{tag}_precision_score": precision,
            f"{tag}_recall_score": recall,
            f"{tag}_f1_score": f1
        }

