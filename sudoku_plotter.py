import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Load and Process CSV Data ---
file_path = "scanned_data.csv"
# file_path = "scanned_data/test/testing/pen_size.csv"
data = pd.read_csv(file_path, header=None)

# Convert values to numeric type and to numpy array
data = data.apply(pd.to_numeric, errors='coerce')
array = data.to_numpy()

# --- Normalize Values to Image ---
image = ((array - np.min(array)) / (np.max(array) - np.min(array)) * 255).astype(np.uint8)

# --- Display Only the Original Image ---
plt.figure(figsize=(5, 5))
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.axis("off")  # Hide axes
plt.show()
