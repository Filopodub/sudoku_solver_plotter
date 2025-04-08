import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Config: Which datasets to include ---
USE_ORIGINAL = True
USE_AUGMENTED = True
USE_AUGMENTED_LIN = True
USE_MNIST = True

# --- Image Properties ---
IMG_SIZE = 28
NUM_CLASSES = 10

# --- Function to Load Custom Dataset ---
def load_data(dataset_path):
    images = []
    labels = []

    for label in range(NUM_CLASSES):
        folder_path = os.path.join(dataset_path, str(label))
        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)
    return images, labels

# --- Load Selected Datasets ---
X_all, y_all = [], []

if USE_ORIGINAL:
    X, y = load_data("training_num_predict/nums_original")
    X_all.append(X)
    y_all.append(y)
    print(f"✅ Loaded Original: {len(X)} samples")

if USE_AUGMENTED:
    X, y = load_data("training_num_predict/nums_augmented")
    X_all.append(X)
    y_all.append(y)
    print(f"✅ Loaded Augmented: {len(X)} samples")

if USE_AUGMENTED_LIN:
    X, y = load_data("training_num_predict/nums_augmented_linear")
    X_all.append(X)
    y_all.append(y)
    print(f"✅ Loaded Augmented linear: {len(X)} samples")

# --- Config: How many MNIST samples to use ---
MNIST_LIMIT = 15000  # Set to None to use all

if USE_MNIST:
    (X_mnist_train, y_mnist_train), (X_mnist_test, y_mnist_test) = tf.keras.datasets.mnist.load_data()
    X_mnist = np.concatenate([X_mnist_train, X_mnist_test], axis=0)
    y_mnist = np.concatenate([y_mnist_train, y_mnist_test], axis=0)

    # Shuffle MNIST
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(X_mnist))

    if MNIST_LIMIT is not None:
        indices = indices[:MNIST_LIMIT]

    X_mnist = X_mnist[indices]
    y_mnist = y_mnist[indices]

    # Normalize and reshape
    X_mnist = X_mnist.astype(np.float32) / 255.0
    X_mnist = X_mnist.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    X_all.append(X_mnist)
    y_all.append(y_mnist)
    print(f"✅ Loaded MNIST: {len(X_mnist)} samples (limited)")


# --- Combine All Selected Data ---
X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

print(f"\n📊 Final Dataset:")
print(f"   Total: {len(X_all)}")
print(f"   Train: {len(X_train)}")
print(f"   Test:  {len(X_test)}")

# --- Build the CNN Model ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# --- Compile the Model ---
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Train the Model ---
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# --- Save the Trained Model ---
model.save("number_predictor_model_augmented_split.h5")

# --- Plot Training Results ---
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
