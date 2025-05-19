import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SVCNN(nn.Module):
    def __init__(self, n_class, pretrained=True):
        super(SVCNN, self).__init__()

        self.net_1 = models.vgg11(pretrained=pretrained).features
        self.net_2 = models.vgg11(pretrained=pretrained).classifier
        self.net_2._modules['6'] = nn.Linear(4096, n_class)

    def forward(self, x):
        y = self.net_1(x)
        y = self.net_2(y.view(y.shape[0],-1))
        return y

class MVCNN(nn.Module):
    def __init__(self, model, n_view=12):
        super(MVCNN, self).__init__()

        self.n_view = n_view
        self.net_1 = model.net_1
        self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.n_view), self.n_view, y.shape[-3], y.shape[-2], y.shape[-1])) #(8,12,512,7,7)
        y = torch.max(y, 1)[0] # max-pooling layer
        y = self.net_2(y.view(y.shape[0],-1))
        return y

