import numpy as np
import cv2
import os
import pandas as pd

def auto_crop(image, threshold=None, max_crop_px=10, percentile=10):
    """
    Auto-crops black borders from a grayscale image, using adaptive threshold and crop limits.

    Args:
        image (np.ndarray): Grayscale image (uint8, range 0–255).
        threshold (int or None): Pixel value threshold. If None, it's calculated using the given percentile.
        max_crop_px (int): Max pixels to crop from each side.
        percentile (float): Percentile used for adaptive threshold if threshold is None (e.g., 10 for darker areas).

    Returns:
        np.ndarray: Cropped image.
    """
    if threshold is None:
        threshold = np.percentile(image, percentile)

    mask = image > threshold
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return image  # skip cropping if everything is dark

    # Crop within safe limits
    top = min(rows[0], max_crop_px)
    bottom = max(rows[-1] + 1, image.shape[0] - max_crop_px)
    left = min(cols[0], max_crop_px)
    right = max(cols[-1] + 1, image.shape[1] - max_crop_px)

    return image[top:bottom, left:right]



def pad_to_square(image, size=28):
    h, w = image.shape
    max_side = max(h, w)
    
    # Use the minimum value from the image as padding
    pad_value = np.max(image)
    padded = np.full((max_side, max_side), pad_value * 0.85, dtype=np.uint8)
    
    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return padded

# --- Define Input and Output Directories ---
training_session = "scanned_data/training3/"
output_folder = "training_num_predict/nums_original"

# Datasets to process (no train/test split now)
datasets = [
    (training_session + "training_data/", list(range(1, 10))),  
    (training_session + "training_data_zeros/", [0] * 9)  
]

# Clear output folders
for digit_value in range(10):
    digit_folder = os.path.join(output_folder, str(digit_value))
    if os.path.exists(digit_folder):
        for file in os.listdir(digit_folder):
            file_path = os.path.join(digit_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared folder: {digit_folder}")

# --- Process Datasets (All go to nums_original) ---
for input_folder, digit_order in datasets:
    os.makedirs(output_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    for csv_file in csv_files:
        csv_path = os.path.join(input_folder, csv_file)
        data = pd.read_csv(csv_path, header=None).values

        data = data[:, 5:370]

        num_digits = 9
        digit_width = data.shape[1] // num_digits
        output_size = 28

        file_base_name = csv_file.replace("scanned_data", "").replace(".csv", "").strip("_")

        for i in range(num_digits):
            digit_value = digit_order[i]

            digit_img = data[:, i * digit_width:(i + 1) * digit_width]
            digit_img = ((digit_img - np.min(digit_img)) / (np.max(digit_img) - np.min(digit_img)) * 255).astype(np.uint8)
            digit_img = auto_crop(digit_img, threshold=None, max_crop_px=5, percentile=90)
            digit_img = pad_to_square(digit_img)
            digit_img = cv2.resize(digit_img, (output_size, output_size), interpolation=cv2.INTER_AREA)


            digit_folder = os.path.join(output_folder, str(digit_value))
            os.makedirs(digit_folder, exist_ok=True)

            save_path = os.path.join(digit_folder, f"{file_base_name}_{i+1}.png")
            cv2.imwrite(save_path, digit_img)

            print(f"Saved: {save_path}")

print("✅ All CSV files processed into nums_original folder!")
