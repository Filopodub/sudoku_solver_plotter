import numpy as np
import cv2
import os
import pandas as pd

# Load CSV data
csv_path = "scanned_data.csv"  # Change to your actual file
data = pd.read_csv(csv_path, header=None).values  # Load as NumPy array

# Ensure data is in 2D form (40x375)
if data.shape != (40, 375):
    raise ValueError("CSV data should have a shape of (40, 375)")

data = data[:, 5:370]  # Crop to the desired range

# Define parameters
num_digits = 9
digit_width = data.shape[1] // num_digits  # ~30 pixels per digit
output_size = 28  # Resize to 28x28 pixels
save_folder = "digits_csv/"  # Output folder

# Ensure output directory exists
os.makedirs(save_folder, exist_ok=True)

# Split and save digits
for i in range(num_digits):
    # Crop each digit from the 275-pixel wide row
    digit_img = data[:, i * digit_width:(i + 1) * digit_width]

    # Normalize values (if necessary) - Convert to 0-255 range
    digit_img = ((digit_img - np.min(digit_img)) / (np.max(digit_img) - np.min(digit_img)) * 255).astype(np.uint8)

    # Resize to 28x28 (for AI training)
    digit_img = cv2.resize(digit_img, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # Save each digit as an image
    save_path = os.path.join(save_folder, f"digit_{i}.png")
    cv2.imwrite(save_path, digit_img)

    print(f"Saved: {save_path}")

print("✅ CSV digits successfully converted into images!")
