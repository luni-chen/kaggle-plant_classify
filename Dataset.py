import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self,root,set,transformer=None):
        self.root=root
        self.set=set
        self.image_path=os.path.join(root,"images")
        self.csv_path=os.path.join(root,"{}.csv".format(set))
        self.transformer=transformer

        # process (id,label)
        labeled_id=pd.read_csv(self.csv_path)
        self.image_ids = labeled_id.iloc[:, 0].to_dict()
        if set=="train":
            self.types = labeled_id.columns.values
            image_labels = np.array(labeled_id.iloc[:, 1:])
            self.image_labels=torch.tensor(image_labels)

    def __getitem__(self, index):
        # return image && label
        img_path=os.path.join(self.image_path,"{}.jpg".format(self.image_ids[index]))
        img=Image.open(img_path).convert("RGB")
        if self.transformer:
            img=self.transformer(img)
            # label = self.image_labels[index]
            # return img, label
        if self.set=="test":
            return img
        elif self.set=="train":
            # label_index=list(self.image_labels[index]).index(1)
            # label=label_index+1
            label=torch.tensor(np.argmax(self.image_labels[index]))
            return img,label

    def __len__(self):
        return len(self.image_ids)