import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from src.utils.image_processing import auto_crop, pad_to_square, preprocess_image, normalize_image

# Constants
MODEL_PATH = "models/trained/number_predictor_model.h5"  # Updated to use the new model path
IMG_SIZE = 28
GRID_SIZE = 9

class SudokuSolver:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.expected_grid = np.array([
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

    def evaluate_sudoku_prediction(self, predicted_array, expected_array):
        """Evaluates accuracy of Sudoku grid prediction."""
        non_zero_mask = expected_array != 0
        correct_count = np.sum((predicted_array == expected_array) & non_zero_mask)
        total_digits = np.count_nonzero(expected_array)
        
        if total_digits == 0:
            return 0.0
            
        accuracy = correct_count / total_digits

        print("\nMismatched positions (row, col):")
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                expected = expected_array[row, col]
                predicted = predicted_array[row, col]
                if expected != 0 and predicted != expected:
                    print(f"({row}, {col}) → expected: {expected}, predicted: {predicted}")

        print(f"\nTotal correct: {correct_count} / {total_digits}")
        print(f"Prediction accuracy: {accuracy:.2%}")
        
        return accuracy

    def process_and_predict(self, csv_path, num_rows=GRID_SIZE):
        """Processes CSV data and predicts Sudoku digits."""
        # Load and preprocess CSV data
        data = pd.read_csv(csv_path, header=None).values
        data = data[:, 5:370]  # Remove unnecessary columns
        
        num_digits = num_rows * GRID_SIZE
        digit_width = data.shape[1] // GRID_SIZE
        
        # Initialize grid and visualization
        sudoku_grid = np.full((num_rows, GRID_SIZE), -1, dtype=int)
        plt.figure(figsize=(10, 10))
        
        total_confidence = 0
        predictions_count = 0

        # Process each digit position
        for i in range(num_digits):
            row = i // GRID_SIZE
            col = i % GRID_SIZE

            # Extract digit image
            row_start = row * (data.shape[0] // num_rows)
            row_end = (row + 1) * (data.shape[0] // num_rows)
            col_start = col * digit_width
            col_end = (col + 1) * digit_width
            
            # Process digit image
            digit_img = data[row_start:row_end, col_start:col_end]
            digit_img = normalize_image(digit_img)
            digit_img = auto_crop(digit_img, max_crop_px=5, percentile=90)
            digit_img = pad_to_square(digit_img)
            digit_img = cv2.resize(digit_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            
            # Predict digit
            processed_image = preprocess_image(digit_img)
            processed_image = processed_image.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # Add channel dimension
            prediction = self.model.predict(processed_image, verbose=0)  # Disable verbose output
            predicted_number = np.argmax(prediction)
            confidence = np.max(prediction)

            total_confidence += confidence
            predictions_count += 1
            sudoku_grid[row, col] = predicted_number

            # Visualize prediction
            plt.subplot(num_rows, GRID_SIZE, i + 1)
            plt.imshow(digit_img, cmap='gray')
            plt.axis("off")
            plt.title(f"{predicted_number}\n{confidence:.2f}")

        plt.tight_layout()
        plt.show()

        # Report average confidence
        if predictions_count > 0:
            mean_confidence = total_confidence / predictions_count
            print(f"Mean confidence: {mean_confidence:.4f}")
        
        return sudoku_grid

def main():
    """Main execution function."""
    solver = SudokuSolver()
    csv_file_path = "data/raw/scanned_data.csv"
    
    sudoku_array = solver.process_and_predict(csv_file_path)
    print("\nRecognized Sudoku Grid:")
    print(sudoku_array)
    
    accuracy = solver.evaluate_sudoku_prediction(sudoku_array, solver.expected_grid)
    return accuracy

if __name__ == "__main__":
    main()
