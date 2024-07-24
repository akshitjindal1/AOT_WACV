#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os.path as osp

import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision.datasets import imagenet
# import natsort
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageNet1k_v2(imagenet):

    def __init__(self, cfg, transform_flag=False):
        root = cfg.THIEF.DATA_ROOT
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://image-net.org/download-images'
            ))
            
        val_transform =  self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
                ])
            
        if transform_flag:
            train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
        else:
            train_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
                ])
            
        train_data = datasets.ImageFolder(root=os.path.join(root, 'train'),
                                      transform = train_transform)
        
        # val_data = datasets.ImageFolder(root=os.path.join(root, 'val'), 
        #                                 transform = val_transform)
        
        return train_data
        
    #     self.root = root
    #     all_images = os.listdir(self.root)
    #     all_images = all_images[:60000]
    #     self.img_paths = natsort.natsorted(all_images)

    #     self.X = []
    #     self.y = []
    #     print("loading thief dataset, will take some time")
    #     for i in tqdm(range(60000)):
    #         img_path = os.path.join(self.root,self.img_paths[i])
    #         image = Image.open(img_path).convert("RGB")
    #         img_tensor = self.transform(image)
    #         self.X.append(img_tensor)
    #         self.y.append([-1])
        
    #     # self.shape = np.array(self.X).shape

    # def __len__(self):
    #     return len(self.img_paths)

    # def __getitem__(self,idx):
    #     return self.X[idx],self.y[idx]
