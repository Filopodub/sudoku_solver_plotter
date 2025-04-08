import numpy as np
import os
import cv2

# Parameters
IMG_SIZE = 28
SHIFT_PIXELS = 2  # how much to shift
source_dir = "training_num_predict/nums_original"
augmented_dir = "training_num_predict/nums_augmented_linear"

# Directions: (dx, dy)
shifts = {
    'up': (0, -SHIFT_PIXELS),
    'down': (0, SHIFT_PIXELS),
    'left': (-SHIFT_PIXELS, 0),
    'right': (SHIFT_PIXELS, 0)
}

# Create output folders
for digit in range(10):
    os.makedirs(os.path.join(augmented_dir, str(digit)), exist_ok=True)

# Apply linear translations
for digit in range(10):
    folder = os.path.join(source_dir, str(digit))
    output_folder = os.path.join(augmented_dir, str(digit))

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            for direction, (dx, dy) in shifts.items():
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted_img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=0)

                new_filename = f"{filename[:-4]}_shift_{direction}.png"
                new_path = os.path.join(output_folder, new_filename)
                cv2.imwrite(new_path, shifted_img)

print("✅ Linear shift augmentations saved!")
