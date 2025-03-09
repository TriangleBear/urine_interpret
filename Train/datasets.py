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
import random  # Add this missing import at the top

# Define class names for reporting
CLASS_NAMES = {
    0: 'Bilirubin', 1: 'Blood', 2: 'Glucose', 3: 'Ketone',
    4: 'Leukocytes', 5: 'Nitrite', 6: 'Protein', 7: 'SpGravity',
    8: 'Urobilinogen', 9: 'Background', 10: 'pH', 11: 'Strip'
}

class UrineStripDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, cache_size=100):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
        
        # Default transform if none provided
        self.class_distribution = {}

        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((512, 512)),  # Change size to 512x512
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Added color jitter
                T.RandomRotation(degrees=15),  # Added random rotation
                T.ToTensor(),

            ])
        
        # Add caching to speed up repeated access
        self.cache = {}
        self.cache_size = cache_size
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Check if in cache first
        if idx in self.cache:
            return self.cache[idx]
        
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # CRITICAL FIX: Read ALL classes from the YOLO file and prioritize them correctly.
        # Ensure that classes 9 (Background) and 11 (Strip) are handled properly.

        all_classes_found = []  # Keep track of all classes found in this file
        
        try:
            if os.path.exists(mask_path) and os.path.getsize(mask_path) > 0:
                with open(mask_path, 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if parts and len(parts) >= 5:  # Valid YOLO format
                            try:
                                class_id = int(parts[0])
                                all_classes_found.append(class_id)
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Error reading YOLO file {mask_path}: {e}")
        
        # FIX: Use correct class prioritization:
        # 1. Reagent pads (0-8, 10) - highest priority
        # 2. Strip (11) - medium priority
        # 3. Background (9) - lowest priority
        
        # Default to background class (9) if no annotations are found.
        # Ensure that classes 9 (Background) and 11 (Strip) are handled properly.

        selected_class = 9  # Background
        
        if all_classes_found:
            # Check for reagent pad classes (0-8, 10) - highest priority.
            # Rotate them based on the sample index to increase diversity
            reagent_pad_classes = [cls for cls in all_classes_found if cls < 9 or cls <= 11]
            if reagent_pad_classes:
                # IMPROVEMENT: Instead of always taking the first reagent pad class,
                # rotate them based on the sample index to increase diversity
                if len(reagent_pad_classes) > 1:
                    # Use sample index to select different classes for different samples
                    selected_class = reagent_pad_classes[idx % len(reagent_pad_classes)]
                else:
                    selected_class = reagent_pad_classes[0]
            # Otherwise, check if we have strip class (11) - medium priority.
            elif 11 in all_classes_found:
                selected_class = 11
        
        # Create segmentation mask based on all annotations and log class distribution

        mask, is_empty_label = self._create_mask_from_yolo(mask_path, debug=False)
        
        # Resize mask to match image size
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Apply transform if provided
        image_tensor = self.transform(image)
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask).long()
        
        # Use our properly selected class as the label
        label = selected_class
        
        if label in self.class_distribution:
            self.class_distribution[label] += 1
        else:
            self.class_distribution[label] = 1
        
        # Add to cache if not full
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (image_tensor, label, self.class_distribution)
        
        # Log class distribution for debugging
        print(f"Class distribution: {self.class_distribution}")
        
        return image_tensor, label, self.class_distribution



    def _create_mask_from_yolo(self, txt_path, image_size=(512, 512), target_classes=None, debug=False):
        """
        Create a segmentation mask from YOLO format annotations.
        Clear prioritization of classes:
        1. Classes 0-8, 10 (reagent pads) - highest priority
        2. Class 11 (strip) - second priority
        3. Class 9 (background) - lowest priority (only fills empty areas)
        """
        # Start with zeros (background/class 9)
        mask = np.zeros(image_size, dtype=np.uint8)
        mask.fill(9)  # Fill with background class explicitly
        is_empty_label = False
        
        # Check if the annotation file exists

        if not os.path.exists(txt_path):
            if debug: print(f"Warning: Missing annotation file: {os.path.basename(txt_path)}")

            is_empty_label = True
            return mask, is_empty_label
        
        # Check if file is empty
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
                # If file is empty or has no valid lines, return empty mask
                if len(lines) == 0 or all(not line.strip() for line in lines):
                    if debug: print(f"Note: Empty annotation file: {os.path.basename(txt_path)}")

                    is_empty_label = True
                    return mask, is_empty_label
                
                # First pass: Group annotations by class
                class_annotations = {i: [] for i in range(NUM_CLASSES)}  # For classes 0-11
                
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        class_annotations[class_id].append(parts)
                    except Exception as e:
                        if debug: print(f"Error processing line: {line.strip()}, Error: {e}")
                
                # Process annotations in Z-order (from back to front, per prioritization)
                
                # First: Background class (9) is already filled as default
                
                # Second: Draw strip (class 11) - overwrites background
                if class_annotations[11]:
                    for parts in class_annotations[11]:
                        try:
                            # Handle different annotation formats
                            if len(parts) > 5:  # Polygon format
                                polygon_points = self._parse_polygon(parts, image_size)
                                if len(polygon_points) >= 3:  # Need at least 3 points for a polygon
                                    cv2.fillPoly(mask, [polygon_points], 11)  # Strip class
                            elif len(parts) == 5:  # Bounding box format
                                x, y, w, h = self._parse_bbox(parts, image_size)
                                cv2.rectangle(mask, (x, y), (x+w, y+h), 11, -1)  # Strip class
                        except Exception as e:
                            if debug: print(f"Error processing strip annotation: {e}")
                
                # Last: Draw reagent pads (classes 0-8, 10) - highest priority, overwrites strip and background
                for class_id in list(range(9)) + [10]:  # 0-8, 10 = reagent pads
                    if class_annotations[class_id]:
                        for parts in class_annotations[class_id]:
                            try:
                                # Handle different annotation formats
                                if len(parts) > 5:  # Polygon format
                                    polygon_points = self._parse_polygon(parts, image_size)
                                    if len(polygon_points) >= 3:  # Need at least 3 points for a polygon
                                        cv2.fillPoly(mask, [polygon_points], class_id)
                                elif len(parts) == 5:  # Bounding box format
                                    x, y, w, h = self._parse_bbox(parts, image_size)
                                    cv2.rectangle(mask, (x, y), (x+w, y+h), class_id, -1)
                            except Exception as e:
                                if debug: print(f"Error processing reagent pad annotation: {e}")
                
        except Exception as e:
            if debug: print(f"Error reading annotation file {txt_path}: {e}")

            is_empty_label = True
            return mask, is_empty_label
        
        # If after processing all lines, mask is still only background (no successful annotations),
        # consider it an empty label
        if np.all(mask == 9):
            is_empty_label = True
            if debug: print(f"Warning: No valid annotations found in {os.path.basename(txt_path)}")
        
        # Optionally filter mask for target classes
        if target_classes is not None:
            mask = np.where(np.isin(mask, target_classes), mask, 9)  # Default to background if not in target classes
        
        return mask, is_empty_label

    # Helper functions to make the code cleaner
    def _parse_polygon(self, parts, image_size):
        points = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= image_size[1]  # Scale x
        points[:, 1] *= image_size[0]  # Scale y
        return points.astype(np.int32)
    
    def _parse_bbox(self, parts, image_size):
        x_center, y_center, width, height = map(float, parts[1:5])
        x = int((x_center - width/2) * image_size[1])
        y = int((y_center - height/2) * image_size[0])
        w = int(width * image_size[1])
        h = int(height * image_size[0])
        return (x, y, w, h)


def visualize_class_distribution(self): 

    import matplotlib.pyplot as plt
    classes = list(self.class_distribution.keys())
    counts = list(self.class_distribution.values())
    
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Class Distribution')
    plt.xticks(classes, [f'Class {cls}' for cls in classes])
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
            transforms.RandomResizedCrop(512, scale=(0.6, 1.0)),  # Change size to 512x512
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.3),  # Adjusted hue range
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # Reduced affine range
            transforms.RandomGrayscale(p=0.3),  # Adjusted grayscale probability
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Adjusted erasing probability
            transforms.Normalize(mean=mean, std=std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST),  # Change size to 512x512
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
            transforms.Resize((512, 512)),  # Change size to 512x512
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST),  # Change size to 512x512
            mask_to_tensor
        ])
    def __call__(self, sample):
        image = self.image_transform(sample['image'])
        mask = Image.fromarray(sample['mask'])  # Convert mask to PIL Image
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()  # Convert mask to tensor
        return {'image': image, 'mask': mask}
