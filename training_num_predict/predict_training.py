import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Load MNIST dataset (handwritten digits 0-9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0-255) to range (0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a simple Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28)),   # Convert 28x28 image to 1D vector
    Dense(128, activation='relu'),   # Hidden layer with 128 neurons
    Dense(64, activation='relu'),    # Another hidden layer
    Dense(10, activation='softmax')  # Output layer (10 classes for digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Function to make predictions on a single image
def predict_number(image_index):
    plt.imshow(x_test[image_index], cmap='gray')  # Show image
    plt.title(f"Actual: {y_test[image_index]}")
    plt.show()
    
    prediction = model.predict(x_test[image_index].reshape(1, 28, 28))  # Predict
    predicted_number = np.argmax(prediction)  # Get most likely class
    print(f"Predicted Number: {predicted_number}")

# Test with a sample image
predict_number(0)  # Change index to test other numbers

# Save trained model
model.save("number_predictor_model.h5")
