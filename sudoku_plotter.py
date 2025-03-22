import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Load and Process CSV Data ---
file_path = "scanned_data/training_data/scanned_data.csv"
data = pd.read_csv(file_path, header=None)

# Convert values to numeric type and to numpy array
data = data.apply(pd.to_numeric, errors='coerce')
array = data.to_numpy()

# --- Normalize Values to Image ---
white_value = 80  
black_value = 10  
normalized_array = np.clip((black_value - array) / (black_value - white_value), 0, 1)
image = (normalized_array * 255).astype(np.uint8)

# --- Sharpening Filter ---
def sharpen_image(img):
    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]]) 
    return cv2.filter2D(img, -1, kernel)

# --- Contrast Enhancement ---
def enhance_contrast(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

# Apply sharpening and contrast enhancement
sharpened = sharpen_image(image)
contrast_enhanced = enhance_contrast(image)

# --- Display Results ---
fig, axs = plt.subplots(3, 1, figsize=(4, 12))  # Arrange vertically

axs[0].imshow(image, cmap="gray", vmin=0, vmax=255)
axs[0].set_title("Original Image")

axs[1].imshow(sharpened, cmap="gray", vmin=0, vmax=255)
axs[1].set_title("Sharpened Image")

axs[2].imshow(contrast_enhanced, cmap="gray", vmin=0, vmax=255)
axs[2].set_title("Contrast Enhanced")

for ax in axs:
    ax.axis("off") 

plt.tight_layout() 
