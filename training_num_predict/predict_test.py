import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model("number_predictor_model.h5")

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
    image = image / 255.0  # Normalize pixel values (0-1)
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

# Function to predict numbers from all images in a dataset folder
def predict_from_folder(folder_path):
    image_paths = []
    
    # Loop through subdirectories (0-9)
    for label in range(10):
        label_path = os.path.join(folder_path, str(label))
        if not os.path.exists(label_path):
            continue  # Skip if folder does not exist
        
        # Get all image file paths in the label folder
        for file in os.listdir(label_path):
            image_paths.append(os.path.join(label_path, file))
    
    # Process and predict for each image
    plt.figure(figsize=(10, 10))
    for i, image_path in enumerate(image_paths):
        image = preprocess_image(image_path)
        prediction = model.predict(image)
        predicted_number = np.argmax(prediction)  # Get the highest probability class
        confidence = np.max(prediction)  # Confidence (highest probability)
        
        # Display the image with its predicted label and accuracy
        plt.subplot(5, 5, i + 1)  # Adjust grid size if needed
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.axis("off")
        plt.title(f"Pred: {predicted_number}\nAcc: {confidence:.2f}")
    
    plt.tight_layout()
    plt.show()

# Example usage: Predict from all images in folder
dataset_path = "training_num_predict/nums_test"  # Replace with your dataset folder
predict_from_folder(dataset_path)
