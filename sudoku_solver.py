import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model("number_predictor_model2.h5")

# Function to preprocess the image for prediction
def preprocess_image(image):
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
    image = image / 255.0  # Normalize pixel values (0-1)
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

# Function to process and predict Sudoku digits from CSV file
def process_and_predict(csv_path, num_rows=9):
    data = pd.read_csv(csv_path, header=None).values  # Load CSV as NumPy array

    # # Check if the data has the correct number of columns
    # if data.shape[1] < 370:
    #     print(f"Skipping {csv_path}: Invalid shape {data.shape}")
    #     return None

    # Crop the relevant part of the data
    data = data[:, 5:370]  # Remove unnecessary columns

    num_digits = num_rows * 9  # Total number of digits to extract
    digit_width = data.shape[1] // 9  # Each digit’s approximate width

    output_size = 28  # Model input size

    # Create an empty 9x9 array filled with -1 (to indicate unrecognized digits)
    sudoku_grid = np.full((num_rows, 9), -1, dtype=int)

    plt.figure(figsize=(10, 10))
    total_confidence = 0
    total_images = 0

    for i in range(num_digits):
        row = i // 9  # Determine row index in the Sudoku grid
        col = i % 9   # Determine column index in the Sudoku grid

        # Extract digit image
        digit_img = data[
            row * (data.shape[0] // num_rows) : (row + 1) * (data.shape[0] // num_rows),
            col * digit_width : (col + 1) * digit_width,
        ]

        # Normalize and process the digit
        digit_img = ((digit_img - np.min(digit_img)) / (np.max(digit_img) - np.min(digit_img)) * 255).astype(np.uint8)
        digit_img = cv2.resize(digit_img, (output_size, output_size), interpolation=cv2.INTER_AREA)
        
        # Predict number
        processed_image = preprocess_image(digit_img)
        prediction = model.predict(processed_image)
        predicted_number = np.argmax(prediction)
        confidence = np.max(prediction)

        total_confidence += confidence
        total_images += 1

        # Store the recognized digit in the Sudoku grid
        sudoku_grid[row, col] = predicted_number

        # Display the image with its prediction
        plt.subplot(num_rows, 9, i + 1)
        plt.imshow(digit_img, cmap='gray')
        plt.axis("off")
        plt.title(f"{predicted_number}\n{confidence:.2f}")

    plt.tight_layout()
    plt.show()

    # Print mean confidence
    if total_images > 0:
        mean_accuracy = total_confidence / total_images
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
    else:
        print("No images to predict.")

    return sudoku_grid  # Return the 9x9 grid as a NumPy array

# Example usage
csv_file_path = "scanned_data.csv"
# csv_file_path = "scanned_data/real_data/scanned_data2.csv";
num_rows = 9  # Change this value if needed (e.g., 3 for partial grid)
sudoku_array = process_and_predict(csv_file_path, num_rows)

print("Recognized Sudoku Grid:")
print(sudoku_array)
