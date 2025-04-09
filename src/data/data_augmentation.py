import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataAugmenter:
    def __init__(self, img_size=28):
        self.img_size = img_size
        self.rotation_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        
    def apply_rotation_augmentation(self, image, num_augmentations=5):
        """Apply rotation-based augmentations."""
        image = image.reshape((1,) + image.shape + (1,))
        augmented_images = []
        
        for _ in range(num_augmentations):
            aug_image = next(self.rotation_datagen.flow(image, batch_size=1))[0]
            augmented_images.append(aug_image.reshape(self.img_size, self.img_size))
            
        return augmented_images
    
    def apply_linear_shifts(self, image, shift_pixels=2):
        """Apply linear shift augmentations."""
        shifts = {
            'up': (0, -shift_pixels),
            'down': (0, shift_pixels),
            'left': (-shift_pixels, 0),
            'right': (shift_pixels, 0)
        }
        
        augmented_images = []
        for direction, (dx, dy) in shifts.items():
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(image, M, (self.img_size, self.img_size), 
                                   borderValue=int(np.max(image) * 0.85))
            augmented_images.append(shifted)
            
        return augmented_images
    
    def augment_dataset(self, source_dir, output_dir, 
                       use_rotation=True, use_linear=True):
        """Augment entire dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        for digit in range(10):
            input_folder = os.path.join(source_dir, str(digit))
            output_folder = os.path.join(output_dir, str(digit))
            os.makedirs(output_folder, exist_ok=True)
            
            if not os.path.exists(input_folder):
                continue
                
            for filename in os.listdir(input_folder):
                if not filename.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(input_folder, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (self.img_size, self.img_size))
                
                # Generate augmentations
                augmented = []
                if use_rotation:
                    augmented.extend(self.apply_rotation_augmentation(image))
                if use_linear:
                    augmented.extend(self.apply_linear_shifts(image))
                
                # Save augmented images
                for idx, aug_image in enumerate(augmented):
                    base_name = os.path.splitext(filename)[0]
                    aug_filename = f"{base_name}_aug_{idx}.png"
                    aug_path = os.path.join(output_folder, aug_filename)
                    cv2.imwrite(aug_path, aug_image)
                    
        print("✅ Dataset augmentation completed!")

def main():
    augmenter = DataAugmenter()
    augmenter.augment_dataset("data/processed/original", "data/processed/augmented")

if __name__ == "__main__":
    main()
