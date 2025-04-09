import numpy as np
import os
import cv2
import tensorflow as tf
from src.utils.image_processing import preprocess_image

class DatasetLoader:
    def __init__(self, img_size=28):
        self.img_size = img_size

    def load_dataset(self, dataset_path):
        """
        Load dataset from specified path.
        
        Args:
            dataset_path (str): Path to dataset directory
            
        Returns:
            tuple: (images, labels) arrays
        """
        images = []
        labels = []

        for label in range(10):  # Assuming 10 classes (0-9)
            folder_path = os.path.join(dataset_path, str(label))
            if not os.path.exists(folder_path):
                continue

            for file in os.listdir(folder_path):
                if not file.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                processed_img = preprocess_image(img)
                images.append(processed_img[0])
                labels.append(label)

        images = np.array(images).reshape(-1, self.img_size, self.img_size, 1)
        labels = np.array(labels)
        return images, labels

    def load_mnist(self, limit=None):
        """
        Load MNIST dataset with optional size limit.
        
        Args:
            limit (int, optional): Maximum number of samples to load
            
        Returns:
            tuple: (images, labels) arrays
        """
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        if limit:
            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(len(X))[:limit]
            X = X[indices]
            y = y[indices]

        X = X.astype(np.float32) / 255.0
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        
        return X, y

