"""
Dataset classes for loading X-ray images and their masks
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class XrayDataset(Dataset):
    """
    Dataset for loading X-ray images and their segmentation masks
    """
    def __init__(self, images_dir, masks_dir, augment=False):
        """
        Args:
            images_dir (str): Directory with X-ray images
            masks_dir (str): Directory with corresponding masks
            augment (bool): Whether to apply data augmentation
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment

        # Get all image files that have corresponding masks
        self.images = []
        self.masks = []

        # List files in the images directory
        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            # Check if image has a corresponding mask
            img_base = os.path.splitext(img_name)[0]
            mask_path = os.path.join(masks_dir, f"{img_base}.png")

            if os.path.exists(mask_path):
                self.images.append(os.path.join(images_dir, img_name))
                self.masks.append(mask_path)

        # Standard transformations
        self.resize_transform = transforms.Resize((512, 512))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            img (torch.Tensor): Preprocessed image
            mask (torch.Tensor): Binary mask
        """
        # Load image and mask
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")  # Grayscale mask

        # Apply transformations
        if self.augment:
            # Apply same transformations to both image and mask
            img, mask = self._apply_augmentation(img, mask)

        # Resize to standard size
        img = self.resize_transform(img)
        mask = self.resize_transform(mask)

        # Convert to tensor
        img = self.to_tensor(img)
        mask = self.to_tensor(mask)

        # Normalize image (but not mask)
        img = self.normalize(img)

        # Binarize mask
        mask = (mask > 0.5).float()

        return img, mask

    def _apply_augmentation(self, img, mask):
        """Apply the same random transformations to both image and mask"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # Random rotation
        angle = random.randint(-30, 30)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)

        # Random brightness and contrast (only for image)
        if random.random() > 0.5:
            brightness_factor = 1.0 + random.uniform(-0.2, 0.2)
            contrast_factor = 1.0 + random.uniform(-0.2, 0.2)
            img = TF.adjust_brightness(img, brightness_factor)
            img = TF.adjust_contrast(img, contrast_factor)

        return img, mask


class CombinedDataset(Dataset):
    """
    Combines multiple datasets into one
    """
    def __init__(self, datasets):
        """
        Args:
            datasets (list): List of Dataset objects
        """
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        # Find the dataset to use
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_lengths) and idx >= self.cumulative_lengths[dataset_idx]:
            dataset_idx += 1

        # Compute the index within the dataset
        if dataset_idx == 0:
            item_idx = idx
        else:
            item_idx = idx - self.cumulative_lengths[dataset_idx - 1]

        # Get the item from the appropriate dataset
        return self.datasets[dataset_idx][item_idx]