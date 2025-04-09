import numpy as np
import cv2
import os
import pandas as pd
from src.utils.image_processing import auto_crop, pad_to_square, normalize_image

def main():
    # Define Input and Output Directories
    input_base = "data/raw"
    output_folder = "data/processed/original"

    # Datasets to process
    datasets = [
        (os.path.join(input_base, "training_data"), list(range(1, 10))),  # 1-9
        (os.path.join(input_base, "training_data_zeros"), [0] * 9)        # All zeros
    ]

    # Clear output folders
    for digit_value in range(10):
        digit_folder = os.path.join(output_folder, str(digit_value))
        if os.path.exists(digit_folder):
            for file in os.listdir(digit_folder):
                file_path = os.path.join(digit_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared folder: {digit_folder}")

    # Process Datasets
    for input_folder, digit_order in datasets:
        os.makedirs(output_folder, exist_ok=True)

        csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

        for csv_file in csv_files:
            csv_path = os.path.join(input_folder, csv_file)
            data = pd.read_csv(csv_path, header=None).values

            data = data[:, 5:370]

            num_digits = 9
            digit_width = data.shape[1] // num_digits
            output_size = 28

            file_base_name = csv_file.replace("scanned_data", "").replace(".csv", "").strip("_")

            for i in range(num_digits):
                digit_value = digit_order[i]

                digit_img = data[:, i * digit_width:(i + 1) * digit_width]
                digit_img = normalize_image(digit_img)
                digit_img = auto_crop(digit_img, threshold=None, max_crop_px=5, percentile=90)
                digit_img = pad_to_square(digit_img)
                digit_img = cv2.resize(digit_img, (output_size, output_size), interpolation=cv2.INTER_AREA)

                digit_folder = os.path.join(output_folder, str(digit_value))
                os.makedirs(digit_folder, exist_ok=True)

                save_path = os.path.join(digit_folder, f"{file_base_name}_{i+1}.png")
                cv2.imwrite(save_path, digit_img)

                print(f"Saved: {save_path}")

    print("✅ All CSV files processed into data/processed/original folder!")

if __name__ == "__main__":
    main()


