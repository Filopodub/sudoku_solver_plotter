import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "scanned_data.csv"  # Change to the path of your file
data = pd.read_csv(file_path, header=None)

# Convert values to numeric type
data = data.apply(pd.to_numeric, errors='coerce')

# Convert to numpy array
array = data.to_numpy()

# Set reference values
white_value = 80  
black_value = 20  

# Normalize values: white_value -> 1 (white), black_value -> 0 (black)
normalized_array = np.clip((black_value - array) / (black_value - white_value), 0, 1)

# Display the matrix as an image
plt.imshow(normalized_array, cmap="gray", vmin=0, vmax=1)
plt.show()
