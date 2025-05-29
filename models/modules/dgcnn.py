import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .classifier import NonLinearClassifier

def _knn(x, k):
    # compute pairwise distance of input points
    # select topk nerest points
    inner = -2*torch.matmul(x.transpose(2, 1), x) 
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_dist = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_dist.topk(k=k, dim=-1)[1] 
    return idx # (batch_size, num_points, k)

def _graph_feature(x, k=20, idx=None):
    # (x: batch_size, dim, n_point)
    batch_size = x.size(0)
    n_point = x.size(2) 
    x = x.view(batch_size, -1, n_point)
    if idx is None:
        idx = _knn(x, k=k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*n_point
    idx = idx + idx_base
    idx = idx.view(-1)
    _, n_dim, _ = x.size() 

    x = x.transpose(2, 1).contiguous() # (B,N,D) -> (B*N, D)
    feature = x.view(batch_size*n_point, -1)[idx, :]
    feature = feature.view(batch_size, n_point, k, n_dim)
    x = x.view(batch_size, n_point, 1, n_dim).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class DGCNN(nn.Module):
    def __init__(self, k, d_embed=512, dropout=0.3, output_channels=40):
        super(DGCNN, self).__init__()

        self.k = k

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.linear = nn.Sequential(
            nn.Linear(2048, d_embed, bias=False),
            nn.BatchNorm1d(d_embed),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout)
        )
        # self.linear2 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Dropout(p=dropout)
        # )
        # self.linear3 = nn.Linear(256, output_channels)
        self.classifier = NonLinearClassifier(
            input_dim=d_embed, 
            num_classes=output_channels, 
            dropout=dropout
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = _graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0] # (n, 64)

        x = _graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # (n, 64)

        x = _graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0] # (n, 64)

        x = _graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0] # (n, 128)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)

        x = torch.cat((x1, x2), dim=1)
        feat = self.linear(x)
        y = self.classifier(feat)
        return y, feat


