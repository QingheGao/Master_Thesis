import torch
import torchvision.datasets as sets
import torchvision.transforms as transforms
import os
import shutil
import numpy as np



def get_mean_std( dir ='./object/training_images',ratio=0.01):
    """Get mean and std by sample ratio
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    trainset = sets.ImageFolder(dir, transform=transform )
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=int(len(trainset)*ratio),
                                             shuffle=True, num_workers=2)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0,2,3))
    std = np.std(train.numpy(), axis=(0,2,3))
    print(mean,std)
    return mean, std


# '/project/qgao/facedata/subsubtrain'
# /project/qgao/hands

def loadsampleface(dirc='/project/qgao/facedata/subsubtrain'):
    # transforms.RandomHorizontalFlip(),
    mean, std = get_mean_std(dirc, ratio=0.01)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        ###change mean
        transforms.Normalize(mean=mean,
                             std=std),
    ])

    trainset = sets.ImageFolder(dirc, transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=1)


    return trainloader

def loadsampleobject(dirc ='/project/qgao/testneurons'):
    # transforms.RandomHorizontalFlip(),
    mean, std = get_mean_std(dirc, ratio=0.01)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        ###change mean
        transforms.Normalize(mean=mean,
                             std=std),
    ])

    trainset = sets.ImageFolder(dirc, transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)


    return trainloader

def loadtraning(batchsize = 100,dirc='/project/qgao/imagenet1000'):
    # transforms.RandomHorizontalFlip(),
    mean, std = get_mean_std(dirc, ratio=0.01)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        ###change mean
        transforms.Normalize(mean=mean,
                             std=std),
    ])

    trainset = sets.ImageFolder(dirc, transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=2)
    return trainloader
