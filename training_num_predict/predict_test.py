import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load the trained model
model = tf.keras.models.load_model("number_predictor_model.h5")

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
    image = image / 255.0  # Normalize pixel values (0-1)
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

# Function to predict a number from an image
def predict_from_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_number = np.argmax(prediction)  # Get the highest probability class
    print(f"Predicted Number: {predicted_number}")

    # Display the image
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_number}")
    plt.show()

# Example usage: Predict from a custom image
image_path = "test_digit3.png"  # Replace with your image file
predict_from_image(image_path)
