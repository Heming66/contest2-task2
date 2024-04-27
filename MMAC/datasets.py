import os
import csv
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch

class MMACDataset(BaseDataset):
    def __init__(
            self,
            lesion,
            augmentation=None,
            preprocessing=None,
    ):
        if lesion == 'LC':
            images_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/1. Images/1. Training Set'
            masks_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/2. Groundtruths/1. Training Set'
        elif lesion == 'CNV':
            images_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/2. Choroidal Neovascularization/1. Images/1. Training Set'
            masks_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/2. Choroidal Neovascularization/2. Groundtruths/1. Training Set'
        elif lesion == 'FS':
            images_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/3. Fuchs Spot/1. Images/1. Training Set'
            masks_dir = './data/2. Segmentation of Myopic Maculopathy Plus Lesions/3. Fuchs Spot/2. Groundtruths/1. Training Set'

        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i])

        mask = cv2.imread(self.masks_fps[i], 0)
        mask = (mask == 255).astype('float')[:,:,None]

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.augmentation == None and self.preprocessing == None:
            return image, mask, os.path.split(self.images_fps[i])[-1]
        
        return image, mask

    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    transform = [
        albu.Resize(height=512,width=512),
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0, rotate_limit=(-180,180), shift_limit=0, p=1, border_mode=0),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.RandomBrightness(limit=0.4),
        albu.RandomContrast(limit=0.4),

    ]
    return albu.Compose(transform)


def get_validation_augmentation():
    return albu.Resize(height=512,width=512)


def to_tensor(x, **kwargs):
    return torch.tensor(x.transpose(2, 0, 1).astype('float32'))


def get_preprocessing(p):


    transform = [
        albu.Lambda(image=p),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(transform)
