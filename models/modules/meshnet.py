import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialDescriptor(nn.Module):

    def __init__(self):
        super(SpatialDescriptor, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ) 

    def forward(self, centers):
        return self.spatial_mlp(centers)
    

class FaceRotateConvolution(nn.Module):

    def __init__(self):
        super(FaceRotateConvolution, self).__init__()

        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
    
    def forward(self, corners):
        feat = (self.rotate_mlp(corners[:, :6]) +
                self.rotate_mlp(corners[:, 3:9])+
                self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))) / 3
        return self.fusion_mlp(feat)
    
class FaceKernelCorrelation(nn.Module):

    def __init__(self, n_kernel=64, sigma=0.2):
        super(FaceKernelCorrelation, self).__init__()

        self.n_kernel = n_kernel
        self.sigma = sigma
        self.weight_alhpa = nn.Parameter(torch.rand(1, n_kernel, 4) * np.pi)
        self.weight_beta = nn.Parameter(torch.rand(1, n_kernel, 4) * 2 * np.pi)
        self.bn = nn.BatchNorm1d(n_kernel)
        self.act = nn.ReLU()

    def forward(self, normals, neighbor_idx):
        b, _, n = normals.size()

        center = normals.unsqueeze(2).expand(-1, -1, self.n_kernel, -1).unsqueeze(4)
        neighbor = torch.gather(normals.unsqueeze(3).expand(-1, -1, -1, 3), 2, neighbor_idx.unsqueeze(1).expand(-1, 3, -1, -1))
        neighbor = neighbor.unsqueeze(2).expand(-1, -1, self.n_kernel, -1, -1)

        feat = torch.cat([center, neighbor], 4)
        feat = feat.unsqueeze(5).expand(-1, -1, -1, -1, -1, 4)
        weight = torch.cat([torch.sin(self.weight_alhpa) * torch.cos(self.weight_beta),
                            torch.sin(self.weight_alhpa) * torch.sin(self.weight_beta),
                            torch.cos(self.weight_alhpa)], 0)
        weight = weight.unsqueeze(0).expand(b, -1, -1, -1)
        weight = weight.unsqueeze(3).expand(-1, -1, -1, n, -1)
        weight = weight.unsqueeze(4).expand(-1, -1, -1, -1, 4, -1)

        dist = torch.sum((feat-weight)**2, 1)
        feat = torch.sum(torch.sum(np.e**(dist/(-2 * self.sigma**2)), 4), 3) / 16

        return self.act(self.bn(feat))

class StrcuturalDescriptor(nn.Module):

    def __init__(self, n_kernel, sigma):
        super(StrcuturalDescriptor, self).__init__()

        self.FRC = FaceRotateConvolution()
        self.FKC = FaceKernelCorrelation(n_kernel, sigma)
        self.strcuctural_mlp = nn.Sequential(
            nn.Conv1d(64+3+n_kernel, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
            nn.Conv1d(131, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU()
        )

    def forward(self, corners, normals, neighbor_idx):
        feat1 = self.FRC(corners)
        feat2 = self.FKC(normals, neighbor_idx)
        return self.strcuctural_mlp(torch.cat([feat1, feat2, normals], 1))
    
class MeshConvolution(nn.Module):

    def __init__(self, aggregation_method, spatial_in, structural_in, spatial_out, structural_out):
        super(MeshConvolution, self).__init__()

        self.spatial_in = spatial_in
        self.structural_in = structural_in
        self.spatial_out = spatial_out
        self.structural_out = structural_out

        assert aggregation_method in ['Concat', 'Max', 'Average']
        self.aggregation_method = aggregation_method

        self.combination_mlp = nn.Sequential(
            nn.Conv1d(self.spatial_in+self.structural_in, self.spatial_out, 1),
            nn.BatchNorm1d(self.spatial_out),
            nn.ReLU()
        )

        if self.aggregation_method == 'Concat':
            self.concat_mlp = nn.Sequential(
                nn.Conv2d(self.structural_in*2, self.structural_in, 1),
                nn.BatchNorm2d(self.structural_in),
                nn.ReLU()
            )

        self.aggregation_mlp = nn.Sequential(
            nn.Conv1d(self.structural_in, self.structural_out, 1),
            nn.BatchNorm1d(self.structural_out),
            nn.ReLU()
        )

    def forward(self, spatial_feat, structural_feat, neighbor_idx):
        b, _, n = spatial_feat.size()

        spatial_feat = self.combination_mlp(torch.cat([spatial_feat, structural_feat], 1))

        if self.aggregation_method == 'Concat':
            structural_feat = torch.cat([structural_feat.unsqueeze(3).expand(-1, -1, -1, 3),
                                    torch.gather(structural_feat.unsqueeze(3).expand(-1, -1, -1, 3), 2, 
                                        neighbor_idx.unsqueeze(1).expand(-1, self.structural_in, -1, -1))], 1)
            structural_feat = self.concat_mlp(structural_feat)
            structural_feat = torch.max(structural_feat, 3)[0]
        elif self.aggregation_method == 'Max':
            structural_feat = torch.cat([structural_feat.unsqueeze(3),
                                    torch.gather(structural_feat.unsqueeze(3).expand(-1, -1, -1, 3), 2, 
                                        neighbor_idx.unsqueeze(1).expand(-1, self.structural_in, -1, -1))], 3)
            structural_feat = torch.max(structural_feat, 3)[0]
        elif self.aggregation_method == 'Average':
            structural_feat = torch.cat([structural_feat.unsqueeze(3),
                                    torch.gather(structural_feat.unsqueeze(3).expand(-1, -1, -1, 3), 2, 
                                        neighbor_idx.unsqueeze(1).expand(-1, self.structural_in, -1, -1))], 3)
            structural_feat = torch.sum(structural_feat, dim=3) / 4
        
        structural_feat = self.aggregation_mlp(structural_feat)
        return spatial_feat, structural_feat
    

class MeshNet(nn.Module):

    def __init__(self, n_kernel, sigma, aggregation_method, require_feat=False, output_channels=40):
        super(MeshNet, self).__init__()

        self.require_feat = require_feat

        self.spatial_descriptor = SpatialDescriptor()
        self.structrual_desciptor = StrcuturalDescriptor(n_kernel, sigma)
        self.mesh_conv1 = MeshConvolution(aggregation_method, 64, 131, 256, 256)
        self.mesh_conv2 = MeshConvolution(aggregation_method, 256, 256, 512, 512)
        
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, output_channels)
        )

    def forward(self, x):
        face = x[..., :15].float()
        neighbor_idx = x[..., 15:].long()

        face = face.permute(0, 2, 1).contiguous()
        centers, corners, normals = face[:, :3], face[:, 3:12], face[:, 12:]
        corners = corners - torch.cat([centers, centers, centers], dim=1)

        spatial_feat0 = self.spatial_descriptor(centers)
        structural_feat0 = self.structrual_desciptor(corners, normals, neighbor_idx)
        spatial_feat1, structural_feat1 = self.mesh_conv1(spatial_feat0, structural_feat0, neighbor_idx)
        spatial_feat2, structural_feat2 = self.mesh_conv2(spatial_feat1, structural_feat1, neighbor_idx)
        spatial_feat3 = self.fusion_mlp(torch.cat([spatial_feat2, structural_feat2], 1))

        feat = self.concat_mlp(torch.cat([spatial_feat1, spatial_feat2, spatial_feat3], 1))
        feat = torch.max(feat, dim=2)[0]
        feat = feat.reshape(feat.size(0), -1)
        feat = self.classifier[:-1](feat)
        y = self.classifier[-1:](feat)

        if self.require_feat:
            return y, feat / torch.norm(feat)
        else:
            return y
