import torch
from torch import nn
from torchvision import models

class Resnet101(nn.Module):

    def __init__(self,num_classes):
        super().__init__()
        resnet=models.resnet101()
        resnet.load_state_dict(torch.load("resnet101-5d3b4d8f.pth"))
        resnet.fc=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(resnet.fc.in_features,out_features=num_classes)
        )
        self.base_model=resnet
        self.softmax=nn.Softmax(dim=1)

    def forward(self,input):
        return self.softmax(self.base_model(input))