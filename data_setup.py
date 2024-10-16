# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:24:08 2024

@author: lshaw
"""
"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import utils
NUM_WORKERS = 0

def create_experiment(experiment='MNIST_binary',batch_size=32, num_workers=NUM_WORKERS,projdim=50):
    if experiment.split('_')[0].lower()=='mnist':
        mnistshape=28*28
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)),
             utils.RandomProjection(projdim,mnistshape)])
        # Create datasets for training & validation, download if necessary
        train_data = datasets.MNIST('./MNIST/train', 
                                                    train=True, transform=transform, download=True)
        test_data = datasets.MNIST('./MNIST/test', 
                                                    train=False, transform=transform, download=True)
        class_names = train_data.classes
        
        if experiment.split('_')[1].lower()=='binary':
            class_names=[item for item in train_data.classes if item[0]=='7' or item[0]=='9']
            ind7=(train_data.targets==7)
            ind9=(train_data.targets==9)
            indices=(ind7+ind9).nonzero().flatten()
            indices=indices[torch.randperm(len(indices))]
            train_data.targets[ind7]=0
            train_data.targets[ind9]=1
            train_data=torch.utils.data.Subset(train_data, indices[:10000])
            ind7=(test_data.targets==7)
            ind9=(test_data.targets==9)
            indices=(ind7+ind9).nonzero().flatten()
            indices=indices[torch.randperm(len(indices))]
            test_data.targets[ind7]=0
            test_data.targets[ind9]=1
            test_data=torch.utils.data.Subset(train_data, indices)

        # Turn images into data loaders
        train_dataloader = DataLoader(
              train_data,
              batch_size=batch_size,
              shuffle=True,
              num_workers=num_workers,
              pin_memory=True)
        
        test_dataloader = DataLoader(
              test_data,
              batch_size=batch_size,
              shuffle=False,
              num_workers=num_workers,
              pin_memory=True)

    return train_dataloader, test_dataloader, class_names