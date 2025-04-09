import numpy as np
import cv2
from typing import List, Tuple, Optional
from src.utils.image_processing import (
    normalize_image, auto_crop, pad_to_square, 
    enhance_contrast, remove_noise, detect_edges
)

class GridProcessor:
    def __init__(self, grid_size: int = 9, img_size: int = 28):
        self.grid_size = grid_size
        self.img_size = img_size

    def extract_cells(self, grid_image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract individual cells from Sudoku grid image."""
        # Enhance image quality
        grid_image = enhance_contrast(grid_image)
        grid_image = remove_noise(grid_image)
        
        height, width = grid_image.shape
        cell_height = height // self.grid_size
        cell_width = width // self.grid_size
        
        cells = []
        positions = []
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Extract cell
                y1 = row * cell_height
                y2 = (row + 1) * cell_height
                x1 = col * cell_width
                x2 = (col + 1) * cell_width
                
                cell = grid_image[y1:y2, x1:x2]
                
                # Process cell
                cell = normalize_image(cell)
                cell = auto_crop(cell, max_crop_px=5, percentile=90)
                cell = pad_to_square(cell)
                cell = cv2.resize(cell, (self.img_size, self.img_size))
                
                cells.append(cell)
                positions.append((row, col))
                
        return cells, positions
    
    def reconstruct_grid(self, predictions: List[int], 
                        positions: List[Tuple[int, int]]) -> np.ndarray:
        """Reconstruct Sudoku grid from predictions."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for pred, (row, col) in zip(predictions, positions):
            grid[row, col] = pred
            
        return grid
    
    def validate_grid(self, grid: np.ndarray) -> bool:
        """Validate Sudoku grid rules."""
        def check_unit(unit):
            unit = unit[unit != 0]
            return len(unit) == len(set(unit))
        
        # Check rows
        for row in grid:
            if not check_unit(row):
                return False
        
        # Check columns
        for col in grid.T:
            if not check_unit(col):
                return False
        
        # Check 3x3 boxes
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = grid[i:i+3, j:j+3].flatten()
                if not check_unit(box):
                    return False
        
        return True

    def find_empty_cell(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find empty cell in grid."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] == 0:
                    return (i, j)
        return None

    def is_valid_move(self, grid: np.ndarray, pos: Tuple[int, int], num: int) -> bool:
        """Check if number placement is valid."""
        row, col = pos
        
        # Check row
        if num in grid[row]:
            return False
        
        # Check column
        if num in grid[:, col]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box = grid[box_row:box_row+3, box_col:box_col+3]
        if num in box:
            return False
        
        return True
