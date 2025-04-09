import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def plot_training_history(self, history):
        """Plot training and validation metrics."""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_cells(self, cells, predictions, confidences):
        """Plot individual cell predictions."""
        rows, cols = 9, 9
        plt.figure(figsize=(15, 15))
        
        for idx, (cell, pred, conf) in enumerate(zip(cells, predictions, confidences)):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(cell, cmap='gray')
            plt.axis('off')
            plt.title(f'{pred}\n{conf:.2f}')
        
        plt.tight_layout()
        plt.show()

    def plot_sudoku_grid(self, grid, title="Sudoku Grid"):
        """Visualize complete Sudoku grid."""
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='Wistia')
        
        for i in range(9):
            for j in range(9):
                plt.text(j, i, str(grid[i, j]), 
                        ha='center', va='center',
                        color='black' if grid[i, j] != 0 else 'gray')
        
        plt.grid(True)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def compare_grids(self, predicted, expected):
        """Compare predicted and expected grids."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Predicted grid
        ax1.imshow(predicted, cmap='Wistia')
        for i in range(9):
            for j in range(9):
                ax1.text(j, i, str(predicted[i, j]), 
                        ha='center', va='center')
        ax1.grid(True)
        ax1.set_title('Predicted Grid')
        ax1.axis('off')
        
        # Expected grid
        ax2.imshow(expected, cmap='Wistia')
        for i in range(9):
            for j in range(9):
                if expected[i, j] != 0:
                    ax2.text(j, i, str(expected[i, j]), 
                            ha='center', va='center')
        ax2.grid(True)
        ax2.set_title('Expected Grid')
        ax2.axis('off')
        
        plt.show()

