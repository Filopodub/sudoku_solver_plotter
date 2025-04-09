import tensorflow as tf

class CNNModel:
    def __init__(self, img_size=28, num_classes=10):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None

    def build(self):
        """Build CNN model architecture."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', 
                                 input_shape=(self.img_size, self.img_size, 1)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model