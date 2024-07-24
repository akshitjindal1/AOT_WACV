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
import imageio
from copy import deepcopy

def imageio_loader(path: str):
    with open(path, "rb") as f:
        img = imageio.imread(f)
        img = img.permute(2,1,0)
        return img

#classToIndex mapping
#classes list
#images,class_index list
class ImageNet1k(ImageFolder):
    test_frac = 0.0

    def __init__(self, cfg, train=True, transform=None, target_transform=None):
        root = cfg.THIEF.DATA_ROOT
        self.cfg = cfg
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://image-net.org/download-images'
            ))
        # if transform:
        #     # self.transform = transforms.Compose([
        #     #         transforms.RandomResizedCrop(224),
        #     #         transforms.RandomHorizontalFlip(),
        #     #         transforms.ToTensor()
        #     #     ])
            
        #     self.transform = transforms.Compose([
        #             transforms.Resize((224, 224)),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor()
        #         ])
            
        # else:
        #     self.transform = transforms.Compose([
        #             transforms.Resize((224,224)),
        #             transforms.ToTensor()
        #         ])
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
        # Initialize ImageFolder
        super().__init__(root, transform=self.transform,
                         target_transform=target_transform)
        self.root = root

        # self.partition_to_idxs = self.get_partition_to_idxs()
        # self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        # self.samples = [self.samples[i] for i in self.pruned_idxs]
        # self.imgs = self.samples
        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))
        self.last_ok = -1
        self.num_corrupt = 0

    def __getitem__(self, index):
        path, target = self.samples[index]
        flag=True
        while(flag):
            try:
                sample = self.loader(path)
                flag=False
                self.last_ok = index
            except:
                import pdb;pdb.set_trace()
                self.num_corrupt = self.num_corrupt + 1
                path,target = self.samples[self.last_ok]
                
        if self.transform is not None:
            sample = self.transform(sample)
        # target = self.oracle(sample.unsqueeze(0).cuda()).argmax(axis=1,keepdims=False)
        # target = target[0]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Note: we perform a 80-20 split of imagenet training
        # While this is not necessary, this is to simply to keep it consistent with the paper
        prev_state = np.random.get_state()
        np.random.seed(self.cfg.DS_SEED)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)

        return partition_to_idxs



class ImageNet1k_soft(ImageFolder):
    test_frac = 0.0

    def __init__(self,cfg,target_model, train=True, transform=False, target_transform=None):
        root = cfg.THIEF.DATA_ROOT
        self.cfg = cfg
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://image-net.org/download-images'
            ))
        if transform:
            self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
                ])
        # Initialize ImageFolder
        super().__init__(root, transform=self.transform,
                         target_transform=target_transform)
        self.root = root

        # self.partition_to_idxs = self.get_partition_to_idxs()
        # self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        # self.samples = [self.samples[i] for i in self.pruned_idxs]
        # self.imgs = self.samples
        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))
        # self.oracle = target_model.eval()
        self.last_ok = -1
        self.num_corrupt = 0
        
        # targets = [e[1] for e in self.samples]
        # targets_copy = copy.deepcopy(targets)
        self.samples = [[e[0],e[1],deepcopy(e[1])] for e in self.samples]
        

    def __getitem__(self, index):
        path, target, target_copy = self.samples[index]
        flag=True
        while(flag):
            try:
                sample = self.loader(path)
                flag=False
                self.last_ok = index
            except:
                import pdb;pdb.set_trace()
                self.num_corrupt = self.num_corrupt + 1
                path,target = self.samples[self.last_ok]
                
        if self.transform is not None:
            sample = self.transform(sample)
        # target = self.oracle(sample.unsqueeze(0).cuda()).argmax(axis=1,keepdims=False)
        # target = target[0]
        if self.target_transform is not None:
            target = self.target_transform(target)
            target_copy = self.target_transform(target_copy)

        return sample, target, target_copy, index

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Note: we perform a 80-20 split of imagenet training
        # While this is not necessary, this is to simply to keep it consistent with the paper
        prev_state = np.random.get_state()
        np.random.seed(self.cfg.DS_SEED)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)

        return partition_to_idxs
