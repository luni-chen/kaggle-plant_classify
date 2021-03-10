import torch
from torch import nn
from torchvision import models

class Resnet50(nn.Module):

    def __init__(self,num_classes):
        super().__init__()
        resnet=models.resnet50()
        resnet.load_state_dict(torch.load("resnet50-19c8e357.pth"))
        resnet.fc=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(resnet.fc.in_features,out_features=num_classes)
        )
        self.base_model=resnet
        self.softmax=nn.Softmax(dim=1)

    def forward(self,input):
        return self.softmax(self.base_model(input))