import torch
from torch import nn
from torchvision import models

class VGG11(nn.Module):

    def __init__(self,num_classes):
        super().__init__()
        vgg=models.vgg11()
        vgg.load_state_dict(torch.load("vgg11-bbd30ac9.pth"))
        vgg.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, num_classes), nn.Softmax(dim=1))
        self.base_model=vgg
        # self.softmax=nn.Softmax(dim=1)

    def forward(self,input):
        # return self.softmax(self.base_model(input))
        return self.base_model(input)