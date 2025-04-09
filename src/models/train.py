import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data.dataset_loader import DatasetLoader
from src.models.cnn_model import CNNModel

class ModelTrainer:
    def __init__(self, img_size=28, num_classes=10):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = CNNModel(img_size, num_classes).build()
        self.dataset_loader = DatasetLoader(img_size)
        
    def train(self, config):
        """
        Train the model with specified datasets.
        
        Args:
            config (dict): Training configuration with keys:
                - use_original (bool): Whether to use original dataset
                - use_augmented (bool): Whether to use augmented dataset
                - use_augmented_lin (bool): Whether to use linear augmented dataset
                - use_mnist (bool): Whether to use MNIST dataset
                - mnist_limit (int): Number of MNIST samples to use (None for all)
                - epochs (int): Number of training epochs
        """
        X_all, y_all = [], []

        # Load original dataset
        if config['use_original']:
            X, y = self.dataset_loader.load_dataset("data/processed/original")
            X_all.append(X)
            y_all.append(y)
            print(f"✅ Loaded Original: {len(X)} samples")

        # Load augmented dataset
        if config['use_augmented']:
            X, y = self.dataset_loader.load_dataset("data/processed/augmented")
            X_all.append(X)
            y_all.append(y)
            print(f"✅ Loaded Augmented: {len(X)} samples")

        # Load linear augmented dataset
        if config['use_augmented_lin']:
            X, y = self.dataset_loader.load_dataset("data/processed/augmented_linear")
            X_all.append(X)
            y_all.append(y)
            print(f"✅ Loaded Augmented linear: {len(X)} samples")

        # Load MNIST dataset
        if config['use_mnist']:
            X, y = self.dataset_loader.load_mnist(limit=config['mnist_limit'])
            X_all.append(X)
            y_all.append(y)
            print(f"✅ Loaded MNIST: {len(X)} samples")

        # Combine all datasets
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )

        print(f"\n📊 Final Dataset:")
        print(f"   Total: {len(X_all)}")
        print(f"   Train: {len(X_train)}")
        print(f"   Test:  {len(X_test)}")

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            validation_data=(X_test, y_test)
        )
        
        self.plot_training_history(history)
        self.save_model()
        
        return history

    def plot_training_history(self, history):
        """Plot training results."""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def save_model(self):
        """Save the trained model."""
        save_path = "models/trained/number_predictor_model.h5"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print("✅ Model saved successfully!")

def main():
    # Training configuration
    config = {
        'use_original': True,
        'use_augmented': True,
        'use_augmented_lin': True,
        'use_mnist': True,
        'mnist_limit': 15000,
        'epochs': 30
    }

    trainer = ModelTrainer()
    trainer.train(config)

if __name__ == "__main__":
    main()
