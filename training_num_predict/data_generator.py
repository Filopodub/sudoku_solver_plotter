from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2

# Parameters
IMG_SIZE = 28
AUGMENTATIONS_PER_IMAGE = 15
source_dir = "training_num_predict/nums_original"
augmented_dir = "training_num_predict/nums_augmented"

# Create destination folders
for digit in range(10):
    os.makedirs(os.path.join(augmented_dir, str(digit)), exist_ok=True)

# Create ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Go through each digit folder
for digit in range(10):
    folder = os.path.join(source_dir, str(digit))
    output_folder = os.path.join(augmented_dir, str(digit))

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.reshape((1, IMG_SIZE, IMG_SIZE, 1)) / 255.0

            # Generate and save augmentations
            count = 0
            for batch in datagen.flow(img, batch_size=1):
                new_filename = f"{filename[:-4]}_aug{count}.png"
                new_path = os.path.join(output_folder, new_filename)

                # Convert float image back to uint8 for saving
                aug_img = (batch[0] * 255).astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE)
                cv2.imwrite(new_path, aug_img)

                count += 1
                if count >= AUGMENTATIONS_PER_IMAGE:
                    break

print("✅ Augmented images saved!")
