import numpy as np
import gzip
import os
import platform
import pickle
import torch
from torch.utils.data import TensorDataset
import torchvision as tv

def getLocalData(dir, name):
    if name == 'mnist':
        train_dataset = tv.datasets.MNIST(dir, train=True, download=True, transform=tv.transforms.ToTensor())
    elif name == 'cifar':
        transform_train = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4), 
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = tv.datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
    elif name == 'har':
        data_x_raw = np.load("./x_train.npy")
        data_x = data_x_raw.reshape(-1, 1, data_x_raw.shape[1],
        data_x_raw.shape[2])
        data_y = np.load("./y_train.npy")
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
    return train_dataset

def getTestData(dir, name):
    if name == 'mnist':
        test_dataset = tv.datasets.MNIST(dir, train=False, transform=tv.transforms.ToTensor())
    elif name == 'cifar':
        transform_test = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_dataset = tv.datasets.CIFAR10(dir, train=False, transform=transform_test)
    elif name == 'har':
        data_x_raw = np.load("./x_test.npy")
        data_x = data_x_raw.reshape(-1, 1, data_x_raw.shape[1],
        data_x_raw.shape[2])
        data_y = np.load("./y_test.npy")
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
    return test_dataset
