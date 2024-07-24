#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets.folder import ImageFolder, default_loader
import imageio


class Indoor67(ImageFolder):
    def __init__(self,cfg, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.VICTIM.DATA_ROOT, 'indoor67')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://web.mit.edu/torralba/www/indoor.html'
            ))

        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'Images'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))
        self.last_ok = None
        self.num_corrupt = 0                                                            

    def __getitem__(self, index):
        path, target = self.samples[index]
        # flag=True
        # while(flag):
        #     try:
        #         sample = self.loader(path)
        #         flag=False
        #         self.last_ok = index
        #     except:
        #         self.num_corrupt = self.num_corrupt + 1
        #         path,target = self.samples[self.last_ok]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # target = self.oracle(sample.unsqueeze(0).cuda()).argmax(axis=1,keepdims=False)
        # target = target[0]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # ----------------- Load list of train images
        test_images = set()
        with open(osp.join(self.root, 'TestImages.txt')) as f:
            for line in f:
                test_images.add(line.strip())

        for idx, (filepath, _) in enumerate(self.samples):
            filepath = filepath.replace(osp.join(self.root, 'Images') + '/', '')
            if filepath in test_images:
                partition_to_idxs['test'].append(idx)
            else:
                partition_to_idxs['train'].append(idx)

        return partition_to_idxs
