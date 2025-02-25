# datasets.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
from config import NUM_CLASSES, IMAGE_SIZE

class UrineStripDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize(IMAGE_SIZE),
                T.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Extract class from filename directly
        # Format could be: class_X.jpg or IMG_XXXX.jpg
        try:
            # First try to extract using underscore pattern (class_X.jpg)
            parts = os.path.splitext(img_name)[0].split('_')
            if len(parts) >= 2 and parts[0].lower() == "class":
                # Format: class_X.jpg
                label = int(parts[1])
            elif len(parts) >= 2 and parts[0] == "IMG":
                # Format: IMG_XXXX.jpg - these appear to be getting misinterpreted
                # For IMG_XXXX files, default to strip (class 10)
                label = 10
            else:
                # Default to strip class (10) if can't extract
                label = 10
        except ValueError:
            # Default to strip class (10) for any parsing errors
            label = 10
        
        # Apply transform if provided
        image_tensor = self.transform(image)
        
        # Ensure label is in valid range
        if label < 0 or label >= NUM_CLASSES:
            # Only print warning if not an IMG_ file (to reduce noise)
            if not img_name.startswith("IMG_"):
                print(f"Warning: Invalid label {label} for {img_name}, defaulting to 10")
            label = 10
        
        # Print the first few samples for debugging
        if idx < 5:
            print(f"Sample {idx}: {img_name} -> Class {label}")
        
        return image_tensor, label

    def _create_mask_from_yolo(self, txt_path, image_size=(256, 256)):
        mask = np.zeros(image_size, dtype=np.uint8)

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    class_id = int(class_id)
                    x = int((x_center - width/2) * image_size[1])
                    y = int((y_center - height/2) * image_size[0])
                    w = int(width * image_size[1])
                    h = int(height * image_size[0])
                    cv2.rectangle(mask, (x, y), (x+w, y+h), class_id, -1)
                elif len(parts) > 5:
                    class_id = int(parts[0])
                    polygon_points = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    polygon_points[:, 0] *= image_size[1]
                    polygon_points[:, 1] *= image_size[0]
                    polygon_points = polygon_points.astype(np.int32)
                    cv2.fillPoly(mask, [polygon_points], class_id)

        # Comment out the mask visualization
        # if self.visualization_count < 5:
        #     plt.imshow(mask, cmap='gray')
        #     plt.title("YOLO Mask")
        #     plt.show()
        #     self.visualization_count += 1
        
        return mask

# Add a function to visualize the dataset
def visualize_dataset(dataset, num_samples=5):
    import matplotlib.pyplot as plt
    for i in range(num_samples):
        image, mask = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        mask = mask.numpy()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis("off")
        
        plt.show()

class RandomFlip:
    def __init__(self, horizontal=True, vertical=False):
        self.horizontal = horizontal
        self.vertical = vertical
    def __call__(self, image):
        if self.horizontal and random.random() > 0.5:
            image = transforms.functional.hflip(image)
        if self.vertical and random.random() > 0.5:
            image = transforms.functional.vflip(image)
        return image

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees
    def __call__(self, image):
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        return image

class RandomAffine:
    def __init__(self, translate=(0.1, 0.1)):
        self.translate = translate
    def __call__(self, image):
        params = transforms.RandomAffine.get_params(
            degrees=[-10, 10],
            translate=self.translate,
            scale_ranges=None,
            shears=None,
            img_size=image.size
        )
        image = transforms.functional.affine(image, angle=params[0], translate=params[1], scale=params[2], shear=params[3])
        return image

def mask_to_tensor(mask):
    return torch.from_numpy(np.array(mask, dtype=np.uint8)).long()

class RandomTrainTransformations:
    def __init__(self, mean, std):
        self.joint_transform = transforms.Compose([
            RandomFlip(horizontal=True, vertical=True),
            RandomRotation(degrees=45),  # Increase rotation range
            RandomAffine(translate=(0.5, 0.5)),  # Increase translation range
        ])
        self.image_transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.4, 1.0)),  # Increase scale range
            transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.5),  # Adjust hue range
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
        image, mask = sample['image'], sample['mask']
        
        # Ensure mask is a PIL Image before applying transforms
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        # Apply joint transformations
        image = self.joint_transform(image)
        mask = self.joint_transform(mask)

        # Apply individual transformations
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

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
        mask = Image.fromarray(sample['mask'])  # Convert mask to PIL Image
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()  # Convert mask to tensor
        return {'image': image, 'mask': mask}
