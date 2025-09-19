"""
Utility functions for image preprocessing
"""

import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms


def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for inference

    Args:
        image_path (str): Path to the image file

    Returns:
        tuple: (preprocessed_tensor, original_image)
    """
    # Load the image
    original_image = cv2.imread(image_path)

    if original_image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert BGR to RGB
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Create PIL Image
    pil_image = Image.fromarray(original_image_rgb)

    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations
    input_tensor = preprocess(pil_image)

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor, original_image


def postprocess_mask(mask, threshold=0.5):
    """
    Post-process a probability mask into a clean binary mask

    Args:
        mask (numpy.ndarray): Probability mask [0-1]
        threshold (float): Threshold for binarization

    Returns:
        numpy.ndarray: Post-processed binary mask
    """
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)

    # Clean up the mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cleaned_mask


def overlay_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    """
    Create a visualization with the mask overlaid on the original image

    Args:
        image (numpy.ndarray): Original image [H, W, 3]
        mask (numpy.ndarray): Binary mask [H, W]
        alpha (float): Transparency factor
        color (tuple): RGB color for the mask overlay

    Returns:
        numpy.ndarray: Image with overlaid mask
    """
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Create a colored mask
    colored_mask = np.zeros_like(image)
    for c in range(3):
        colored_mask[:, :, c] = binary_mask * color[c]

    # Create a visualization with the mask overlaid on the image
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    # Create contour around the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay