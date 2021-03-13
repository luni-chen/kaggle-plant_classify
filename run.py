from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from resnet50 import Resnet50
from resnet101 import Resnet101
from vgg11 import VGG11
from Dataset import MyDataset
import torch
from torch import nn,optim
import pandas as pd
import numpy as np
import cv2
from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    RandomBrightness,
    RandomContrast,
    Resize,
    ShiftScaleRotate,
    VerticalFlip,
)




# image_size=(448,448)
image_size=(545,545)
batch_size=32
num_classes=4
lr=0.01
momentum=0.9
num_epoch=4
root="/data/panchen2/Desktop/plant_classify/plant-pathology-2020-fgvc7"
#设置数据标准化的均值和标准差
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model=Resnet50(num_classes=num_classes)
# model=Resnet101(num_classes=num_classes)
model=VGG11(num_classes=num_classes)
if torch.cuda.device_count()>1:
    print("Let's use",torch.cuda.device_count(),"GPUs!")
    model=nn.DataParallel(model)
model.to(device=device)

train_transform=transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean,std=norm_std)
])
# train_transform = Compose(
#         [
#             Resize(height=image_size[0], width=image_size[1]),
#             OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
#             OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
#             VerticalFlip(p=0.5),
#             HorizontalFlip(p=0.5),
#             ShiftScaleRotate(
#                 shift_limit=0.2,
#                 scale_limit=0.2,
#                 rotate_limit=20,
#                 interpolation=cv2.INTER_LINEAR,
#                 border_mode=cv2.BORDER_REFLECT_101,
#                 p=1,
#             ),
#             Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
#         ]
#     )
test_transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean,std=norm_std)
])

train_data=MyDataset(root,"train",train_transform)
test_data=MyDataset(root,"test",test_transform)

train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=batch_size,shuffle=False)

criterion=nn.CrossEntropyLoss()
# criterion=nn.MultiLabelMarginLoss()
optimizer=optim.SGD(model.parameters(),lr=lr,momentum=momentum)

print("Start Training!")

for epoch in range(num_epoch):
    # loss_count = 0.0
    for i, data in enumerate(train_dataloader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        # loss_count += loss.item()
        print("epoch: {}, batch: {}, loss: {}".format(
            epoch+1,i+1,loss.data
        ))
        # print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, loss_count / 2))
        # loss_count=0.0
print("End Training!")
torch.save(model,"myResNet50.pth")

# model=torch.load("myResNet50.pth")
# model.to(device)

print("Start Testing!")
predicts=torch.rand(32,4)
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        data.to(device)
        print("batch: {}".format(i + 1), np.shape(data))
        predict = model(data)
        print("batch: {}".format(i + 1), np.shape(predict))
        if i == 0:
            predicts = predict.cpu()
        else:
            predicts = torch.cat((predicts, predict.cpu()), 0)

test=pd.read_csv("plant-pathology-2020-fgvc7/test.csv")
test.columns=["image_id"]
types=list(train_data.types)
del types[0]
submission=pd.concat([test,pd.DataFrame(np.array(predicts).
            reshape(-1,4),columns=types)],axis=1)
submission.reset_index(drop=True,inplace=True)
submission.to_csv("submissions.csv",index=False)

