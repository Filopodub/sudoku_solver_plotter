import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

# --- Load the Trained Model ---
model = tf.keras.models.load_model("number_predictor_model.h5")

# --- Function to Preprocess Image for Prediction ---
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  #
    image = cv2.resize(image, (28, 28))  
    image = image / 255.0  
    image = image.reshape(1, 28, 28)  
    return image

# --- Function to Predict Numbers from Images in Folder ---
def predict_from_folder(folder_path):
    image_paths = []
    total_confidence = 0 
    total_images = 0  
    
    # Loop through subdirectories (0-9)
    for label in range(10):
        label_path = os.path.join(folder_path, str(label))
        if not os.path.exists(label_path):
            continue  
        
        # Get all image file paths in the label folder
        for file in os.listdir(label_path):
            image_paths.append(os.path.join(label_path, file))
    
    # Process and predict for each image
    plt.figure(figsize=(10, 10))
    for i, image_path in enumerate(image_paths):
        image = preprocess_image(image_path)
        prediction = model.predict(image)
        predicted_number = np.argmax(prediction)  
        confidence = np.max(prediction) 
        
        # Accumulate confidence and count images
        total_confidence += confidence
        total_images += 1
        
        # Display the image with its predicted label and accuracy
        plt.subplot(5, 5, i + 1)  
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.axis("off")
        plt.title(f"Pred: {predicted_number}\nAcc: {confidence:.2f}")
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print the mean accuracy (mean confidence)
    if total_images > 0:
        mean_accuracy = total_confidence / total_images
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
    else:
        print("No images to predict.")

# --- Example Usage ---
dataset_path = "training_num_predict/nums_test"  
predict_from_folder(dataset_path)
