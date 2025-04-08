import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model("number_predictor_model_augmented_split.h5")

wanted_array = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

def auto_crop(image, threshold=None, max_crop_px=10, percentile=10):
    """
    Auto-crops black borders from a grayscale image, using adaptive threshold and crop limits.

    Args:
        image (np.ndarray): Grayscale image (uint8, range 0–255).
        threshold (int or None): Pixel value threshold. If None, it's calculated using the given percentile.
        max_crop_px (int): Max pixels to crop from each side.
        percentile (float): Percentile used for adaptive threshold if threshold is None (e.g., 10 for darker areas).

    Returns:
        np.ndarray: Cropped image.
    """
    if threshold is None:
        threshold = np.percentile(image, percentile)

    mask = image > threshold
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return image  # skip cropping if everything is dark

    # Crop within safe limits
    top = min(rows[0], max_crop_px)
    bottom = max(rows[-1] + 1, image.shape[0] - max_crop_px)
    left = min(cols[0], max_crop_px)
    right = max(cols[-1] + 1, image.shape[1] - max_crop_px)

    return image[top:bottom, left:right]



def pad_to_square(image, size=28):
    h, w = image.shape
    max_side = max(h, w)
    
    # Use the minimum value from the image as padding
    pad_value = np.max(image)
    padded = np.full((max_side, max_side), pad_value * 0.85, dtype=np.uint8)
    
    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return padded



def evaluate_sudoku_prediction(predicted_array, expected_array):
    """
    Compare the predicted Sudoku grid with the expected (ground truth) grid.

    Args:
        predicted_array (np.ndarray): The 9x9 predicted Sudoku grid.
        expected_array (np.ndarray): The 9x9 expected Sudoku grid (non-zero where digits are known).

    Returns:
        float: Accuracy of the prediction (non-zero digits only).
    """
    non_zero_mask = expected_array != 0
    correct_count = np.sum((predicted_array == expected_array) & non_zero_mask)
    total_digits = np.count_nonzero(expected_array)
    accuracy = correct_count / total_digits if total_digits > 0 else 0

    print("\nMismatched positions (row, col):")
    for row in range(9):
        for col in range(9):
            expected = expected_array[row, col]
            predicted = predicted_array[row, col]
            if expected != 0 and predicted != expected:
                print(f"({row}, {col}) → expected: {expected}, predicted: {predicted}")

    print(f"\nTotal correct: {correct_count} / {total_digits}")
    print(f"Prediction accuracy (non-zero digits only): {accuracy:.2%}")
    
    return accuracy


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

        # Normalize to 0–255 range
        digit_img = ((digit_img - np.min(digit_img)) / (np.max(digit_img) - np.min(digit_img)) * 255).astype(np.uint8)

        digit_img = auto_crop(digit_img, threshold=None, max_crop_px=5, percentile=90)
        digit_img = pad_to_square(digit_img)
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

accuracy = evaluate_sudoku_prediction(sudoku_array, wanted_array)

