import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Image properties
IMG_SIZE = 28  # Resize images to 28x28
NUM_CLASSES = 10  # Digits 0-9

# Function to load images from a folder
def load_data(dataset_path):
    images = []
    labels = []
    
    for label in range(NUM_CLASSES):
        folder_path = os.path.join(dataset_path, str(label))
        if not os.path.exists(folder_path):  
            continue  # Skip if folder does not exist
        
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
            img = img / 255.0  # Normalize pixel values (0-1)
            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for CNN
    labels = np.array(labels)

    return images, labels

# Load training and testing data separately
train_path = "training_num_predict/nums_test"
test_path = "training_num_predict/nums_test"

X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)

# Build a CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the trained model
model.save("number_predictor_model.h5")

# Plot training results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
