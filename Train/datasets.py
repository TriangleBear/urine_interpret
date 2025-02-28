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
        self.class_distribution = {}

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
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load mask - check if file exists and has content
        mask, is_empty_label = self._create_mask_from_yolo(mask_path)
        
        # Resize mask to match image size
        mask = cv2.resize(mask, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        # Apply transform if provided
        image_tensor = self.transform(image)

        # Convert mask to tensor and add channel dimension
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).long()  # Ensure mask has a channel dimension

        # Extract class label from mask
        # IMPORTANT: We're not using class 0 for empty labels anymore!
        if is_empty_label:
            # Use a special value for empty labels (NUM_CLASSES = 11)
            # This separates empty labels from actual class 0 (Bilirubin)
            label = NUM_CLASSES  # Using NUM_CLASSES (11) as the empty/background indicator
        else:
            label = torch.unique(mask_tensor)
            if len(label) == 1:
                label = label.item()
            else:
                label = 10  # Default to strip class if multiple labels are found
        
        # Update class distribution
        if isinstance(label, int):  # Check if label is already an integer
            if label in self.class_distribution:
                self.class_distribution[label] += 1
            else:
                self.class_distribution[label] = 1

        return image_tensor, label, self.class_distribution


    def _create_mask_from_yolo(self, txt_path, image_size=(256, 256), target_classes=None):
        """
        Create a segmentation mask from YOLO format annotations.
        Respects class hierarchy: Classes 0-9 (pads) > Class 10 (strip) > Class 11 (background)
        """
        mask = np.zeros(image_size, dtype=np.uint8)  # Start with zeros (background/class 11)
        is_empty_label = False
        
        # Check if file exists
        if not os.path.exists(txt_path):
            print(f"Warning: Missing label file: {os.path.basename(txt_path)}")
            is_empty_label = True
            return mask, is_empty_label
        
        # Check if file is empty
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
                # If file is empty or has no valid lines, return empty mask
                if len(lines) == 0 or all(not line.strip() for line in lines):
                    print(f"Note: Empty label file: {os.path.basename(txt_path)}")
                    is_empty_label = True
                    return mask, is_empty_label
                
                # First pass: Group annotations by class
                class_annotations = {i: [] for i in range(12)}  # For classes 0-11
                
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        class_annotations[class_id].append(parts)
                    except Exception as e:
                        print(f"Error processing line: {line.strip()}, Error: {e}")
                
                # Process annotations in Z-order (from back to front)
                # Background (11) is already set as zeros
                
                # Process strip (class 10) first
                if class_annotations[10]:
                    for parts in class_annotations[10]:
                        class_id = 10
                        try:
                            # Handle different annotation formats
                            if len(parts) > 5:  # Polygon format
                                polygon_points = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                                polygon_points[:, 0] *= image_size[1]
                                polygon_points[:, 1] *= image_size[0]
                                polygon_points = polygon_points.astype(np.int32)
                                cv2.fillPoly(mask, [polygon_points], class_id)
                            elif len(parts) == 5:  # Bounding box format
                                x_center, y_center, width, height = map(float, parts[1:5])
                                x = int((x_center - width/2) * image_size[1])
                                y = int((y_center - height/2) * image_size[0])
                                w = int(width * image_size[1])
                                h = int(height * image_size[0])
                                cv2.rectangle(mask, (x, y), (x+w, y+h), class_id, -1)
                        except Exception as e:
                            print(f"Error processing strip annotation: {e}")
                
                # Then process reagent pads (classes 0-9) to overlay on top
                for class_id in range(10):  # 0-9 reagent pads
                    if class_annotations[class_id]:
                        for parts in class_annotations[class_id]:
                            try:
                                # Handle different annotation formats
                                if len(parts) > 5:  # Polygon format
                                    polygon_points = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                                    polygon_points[:, 0] *= image_size[1]
                                    polygon_points[:, 1] *= image_size[0]
                                    polygon_points = polygon_points.astype(np.int32)
                                    cv2.fillPoly(mask, [polygon_points], class_id)
                                elif len(parts) == 5:  # Bounding box format
                                    x_center, y_center, width, height = map(float, parts[1:5])
                                    x = int((x_center - width/2) * image_size[1])
                                    y = int((y_center - height/2) * image_size[0])
                                    w = int(width * image_size[1])
                                    h = int(height * image_size[0])
                                    cv2.rectangle(mask, (x, y), (x+w, y+h), class_id, -1)
                            except Exception as e:
                                print(f"Error processing reagent pad annotation: {e}")
                
        except Exception as e:
            print(f"Error reading label file {txt_path}: {e}")
            is_empty_label = True
            return mask, is_empty_label
        
        # If after processing all lines, mask is still empty (no successful annotations),
        # consider it an empty label
        if np.all(mask == 0):
            is_empty_label = True
            print(f"Warning: No valid annotations found in {os.path.basename(txt_path)}")
        
        # Optionally filter mask for target classes
        if target_classes is not None:
            mask = np.where(np.isin(mask, target_classes), mask, 0)
        
        return mask, is_empty_label


# Add a function to visualize the dataset
def visualize_class_distribution(class_distribution):
    import matplotlib.pyplot as plt
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())
    
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Class Distribution')
    plt.show()

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
            RandomRotation(degrees=30),  # Reduced rotation range
            RandomAffine(translate=(0.1, 0.1)),  # Reduced translation range
        ])
        self.image_transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.6, 1.0)),  # Adjusted scale range
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.3),  # Adjusted hue range
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # Reduced affine range
            transforms.RandomGrayscale(p=0.3),  # Adjusted grayscale probability
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Adjusted erasing probability
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
