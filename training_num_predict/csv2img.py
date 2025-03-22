import numpy as np
import cv2
import os
import pandas as pd

# Define input and output directories
datasets = [
    ("scanned_data/training_data/", "training_num_predict/nums_train/", list(range(1, 10))),  
    ("scanned_data/testing_data/", "training_num_predict/nums_test/", list(range(1, 10))), 
    ("scanned_data/training_data_zeros/", "training_num_predict/nums_train/", [0] * 9)  
]

# Process each dataset (training & testing)
for input_folder, output_folder, digit_order in datasets:
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    # Clear the digit folders before processing new data
    for digit_value in range(10):  # For each digit (0-9)
        digit_folder = os.path.join(output_folder, str(digit_value))

        # Remove all images in the digit folder if it exists
        if os.path.exists(digit_folder):
            for file in os.listdir(digit_folder):
                file_path = os.path.join(digit_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    # Process each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(input_folder, csv_file)
        data = pd.read_csv(csv_path, header=None).values  # Load as NumPy array

        # Ensure data is in correct shape
        if data.shape != (46, 375):
            print(f"Skipping {csv_file}: Invalid shape {data.shape}")
            continue

        data = data[:, 5:370]  # Crop to the desired range

        # Define parameters
        num_digits = 9
        digit_width = data.shape[1] // num_digits  # ~30 pixels per digit
        output_size = 28  # Resize to 28x28 pixels

        # Remove "scanned_data" from filename and remove ".csv"
        file_base_name = csv_file.replace("scanned_data", "").replace(".csv", "").strip("_")

        # Split and save digits
        for i in range(num_digits):
            digit_value = digit_order[i]  # Get correct digit value from order

            digit_img = data[:, i * digit_width:(i + 1) * digit_width]

            # Normalize values (Convert to 0-255 range)
            digit_img = ((digit_img - np.min(digit_img)) / (np.max(digit_img) - np.min(digit_img)) * 255).astype(np.uint8)

            # Resize to 28x28 (for AI training)
            digit_img = cv2.resize(digit_img, (output_size, output_size), interpolation=cv2.INTER_AREA)

            # Create a subfolder for each digit (e.g., 5, 3, 0, etc.)
            digit_folder = os.path.join(output_folder, str(digit_value))

            # Save each digit as an image
            save_path = os.path.join(digit_folder, f"{file_base_name}_{i+1}.png")
            cv2.imwrite(save_path, digit_img)

            print(f"Saved: {save_path}")

print("✅ All CSV files processed and converted into images for both datasets!")
