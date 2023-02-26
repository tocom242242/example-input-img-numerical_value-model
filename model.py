import torch
import torchvision.models as models
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.mlp = nn.Sequential(
            nn.Linear(1, 30), nn.BatchNorm1d(30), nn.ReLU()
        )
        self.output_layer = nn.Linear(1000 + 30, 1)

    def forward(self, x, meta):
        z = self.resnet(x)
        z2 = self.mlp(meta)
        z3 = torch.cat((z, z2), dim=1)
        output = self.output_layer(z3)
        return output