import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

# 1. Define Paths & Parameters
DATA_DIR = r"C:\Users\bamilosin\Documents\500L\coursework\cpe503 - embedded sys\cfd\cfd\CFD Version 3.0.1\Images\CFD-MR"
IMG_SIZE = (256, 256)  # Resizing ensures structural uniformity

# LBP Hyperparameters
RADIUS = 3
N_POINTS = 8 * RADIUS  # 24 neighborhood points
METHOD = "uniform"
N_BINS = N_POINTS + 2  # 'uniform' method yields exactly n_points + 2 bins

def preprocess_and_extract_lbp(image_path, target_size=IMG_SIZE):
    """Loads, converts to gray, resizes, and extracts normalized LBP histogram."""
    # Read the image (handles both RGB or Grayscale paths safely)
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to standardize processing size across the dataset
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Extract LBP map
    lbp = local_binary_pattern(resized, N_POINTS, RADIUS, method=METHOD)
    
    # Calculate the normalized histogram (so pixel count doesn't alter features)
    hist, _ = np.histogram(lbp.ravel(), bins=N_BINS, range=(0, N_BINS), density=True)
    return hist

# 2. Process the Dataset
features_list = []
labels_list = []
filenames_list = []

print(f"Scanning directory: {DATA_DIR}...")

# Walk through the folder (supports flat directory or subdirectories/classes)
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        # Process only common image formats
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(root, file)
            
            # Use the folder name directly above the image as the target class label
            class_label = os.path.basename(root) if root != DATA_DIR else "unknown"
            
            hist_features = preprocess_and_extract_lbp(file_path)
            
            if hist_features is not None:
                features_list.append(hist_features)
                labels_list.append(class_label)
                filenames_list.append(file)

# 3. Convert to Structured Formats
X = np.array(features_list)
y = np.array(labels_list)

print(f"\nProcessing Complete!")
print(f"Successfully processed {len(X)} images.")
print(f"Feature matrix shape: {X.shape} (Images, LBP bins)")

# Optional: Save features to a CSV file for quick loading during classification
df = pd.DataFrame(X)
df['label'] = y
df['filename'] = filenames_list
output_csv = "lbp_extracted_features.csv"
df.to_csv(output_csv, index=False)
print(f"Features saved successfully to: {output_csv}")
