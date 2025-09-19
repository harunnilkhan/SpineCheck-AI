import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil

# Dizin yapısı
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPINE_DATASET_DIR = os.path.join(ROOT_DIR, "spine_dataset")
MASK_OUTPUT_DIR = os.path.join(ROOT_DIR, "dataset/masks")
RAW_OUTPUT_DIR = os.path.join(ROOT_DIR, "dataset/raw")
SETS = ["train", "validation", "test"]
TARGET_SIZE = (512, 512)  # Hedef boyut: 512x512

def load_image_size(image_path):
    img = cv2.imread(image_path)
    return img.shape[1], img.shape[0]  # width, height

def normalized_to_pixel_coords(coords, img_width, img_height):
    return [(int(float(x) * img_width), int(float(y) * img_height)) for x, y in zip(coords[::2], coords[1::2])]

def generate_mask(image_path, label_path, output_mask_path, output_raw_path):
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:  # class + 4 points (8 coords) minimum
            continue
        coords = list(map(float, parts[1:]))
        polygon = normalized_to_pixel_coords(coords, img_width, img_height)
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    # Görüntü ve maskı 512x512'ye yeniden boyutlandır
    resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)

    cv2.imwrite(output_mask_path, resized_mask)
    cv2.imwrite(output_raw_path, resized_img)  # Artık kopyalamak yerine yeniden boyutlandırılmış görüntüyü kaydediyoruz

def process_dataset():
    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)

    for subset in SETS:
        image_dir = os.path.join(SPINE_DATASET_DIR, subset, "images")
        label_dir = os.path.join(SPINE_DATASET_DIR, subset, "labels")
        image_paths = glob(os.path.join(image_dir, "*.jpg"))

        for image_path in tqdm(image_paths, desc=f"Processing {subset}"):
            filename = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, filename + ".txt")
            output_mask_path = os.path.join(MASK_OUTPUT_DIR, filename + ".png")
            output_raw_path = os.path.join(RAW_OUTPUT_DIR, filename + ".jpg")

            if os.path.exists(label_path):
                generate_mask(image_path, label_path, output_mask_path, output_raw_path)

if __name__ == "__main__":
    process_dataset()