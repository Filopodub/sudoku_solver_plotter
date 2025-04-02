import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import shutil
import random

# --- Load the Trained Model ---
model = tf.keras.models.load_model("number_predictor_model.h5")

# --- Function to Preprocess Image for Prediction ---
def preprocess_image(image):
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
    image = image / 255.0  # Normalize pixel values (0-1)
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

# --- Function to Process and Predict from a CSV File ---
def process_and_predict(csv_path):
    data = pd.read_csv(csv_path, header=None).values  # Load CSV as NumPy array
    
    if data.shape != (40, 375):
        print(f"Skipping {csv_path}: Invalid shape {data.shape}")
        return

    data = data[:, 5:370]  # Crop to the desired range
    num_digits = 9
    digit_width = data.shape[1] // num_digits  # ~30 pixels per digit
    output_size = 28  # Resize to 28x28 pixels

    plt.figure(figsize=(10, 10))
    total_confidence = 0
    total_images = 0

    for i in range(num_digits):
        digit_img = data[:, i * digit_width:(i + 1) * digit_width]
        digit_img = ((digit_img - np.min(digit_img)) / (np.max(digit_img) - np.min(digit_img)) * 255).astype(np.uint8)
        digit_img = cv2.resize(digit_img, (output_size, output_size), interpolation=cv2.INTER_AREA)
        
        # Predict number
        processed_image = preprocess_image(digit_img)
        prediction = model.predict(processed_image)
        predicted_number = np.argmax(prediction)
        confidence = np.max(prediction)
        
        total_confidence += confidence
        total_images += 1

        # Display the image with its predicted label and accuracy
        plt.subplot(3, 3, i + 1)
        plt.imshow(digit_img, cmap='gray')
        plt.axis("off")
        plt.title(f"Pred: {predicted_number}\nAcc: {confidence:.2f}")
    
    plt.tight_layout()
    plt.show()
    
    if total_images > 0:
        mean_accuracy = total_confidence / total_images
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
    else:
        print("No images to predict.")

# --- Example Usage ---
csv_file_path = "scanned_data.csv"
# csv_file_path = "scanned_data/real_training_data/scanned_data2.csv" 
process_and_predict(csv_file_path)

