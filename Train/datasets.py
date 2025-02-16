# datasets.py
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2

IMAGE_SIZE = (256, 256)

class UrineStripDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.txt_files = sorted(os.listdir(mask_folder))
        self.transform = transform
        self.image_size = IMAGE_SIZE  # Use the new image size
        
        if len(self.image_files) != len(self.txt_files):
            raise ValueError("Mismatch between number of images and masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        txt_path = os.path.join(self.mask_folder, self.txt_files[idx])
        
        image = Image.open(image_path).convert("RGB").resize(self.image_size)
        mask = self._create_mask_from_yolo(txt_path, image_size=self.image_size)
        
        if self.transform:
            sample = self.transform({'image': image, 'mask': mask})
            return sample['image'], sample['mask']
            
        return transforms.ToTensor()(image), torch.from_numpy(mask).long()

    def _create_mask_from_yolo(self, txt_path, image_size=(256, 256)):
        mask = np.zeros(image_size, dtype=np.uint8)
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    # Convert YOLO format to bounding box coordinates
                    x = int((x_center - width/2) * image_size[1])
                    y = int((y_center - height/2) * image_size[0])
                    w = int(width * image_size[1])
                    h = int(height * image_size[0])
                    cv2.rectangle(mask, (x, y), (x+w, y+h), int(class_id), -1)
        return mask

class RandomFlip:
    def __init__(self, horizontal=True, vertical=False):
        self.horizontal = horizontal
        self.vertical = vertical
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if self.horizontal and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if self.vertical and random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        return {'image': image, 'mask': mask}

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        return {'image': image, 'mask': mask}

class RandomAffine:
    def __init__(self, translate=(0.1, 0.1)):
        self.translate = translate
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        params = transforms.RandomAffine.get_params(
            degrees=[-10, 10],
            translate=self.translate,
            scale_ranges=None,
            shears=None,
            img_size=image.size
        )
        image = transforms.functional.affine(image, angle=params[0], translate=params[1], scale=params[2], shear=params[3])
        mask = transforms.functional.affine(mask, angle=params[0], translate=params[1], scale=params[2], shear=params[3])
        return {'image': image, 'mask': mask}

class RandomTrainTransformations:
    def __init__(self, mean, std):
        self.joint_transform = transforms.Compose([
            RandomFlip(horizontal=True, vertical=True),
            RandomRotation(degrees=45),  # Increase rotation range
            RandomAffine(translate=(0.5, 0.5))  # Increase translation range
        ])
        self.image_transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.4, 1.0)),  # Increase scale range
            transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.6),  # Increase jitter range
            transforms.RandomAffine(degrees=120, translate=(0.5, 0.5), scale=(0.4, 1.6), shear=40),  # Increase affine range
            transforms.RandomGrayscale(p=0.5),  # Increase grayscale probability
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.6, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Increase erasing probability
            transforms.Normalize(mean=mean, std=std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            mask_to_tensor
        ])
    def __call__(self, sample):
        sample = self.joint_transform(sample)
        image = self.image_transform(sample['image'])
        mask = self.mask_transform(sample['mask'])
        return {'image': image, 'mask': mask}

class SimpleValTransformations:
    def __init__(self, mean, std):
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            mask_to_tensor
        ])
    def __call__(self, sample):
        image = self.image_transform(sample['image'])
        mask = self.mask_transform(sample['mask'])
        return {'image': image, 'mask': mask}