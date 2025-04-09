import numpy as np
import cv2

def preprocess_image(image, target_size=28):
    """Preprocesses image for model prediction.
    
    Args:
        image (np.ndarray): Input image
        target_size (int, optional): Target size for resizing. Defaults to 28.
    
    Returns:
        np.ndarray: Preprocessed image ready for model input
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Resize to target size
    image = cv2.resize(image, (target_size, target_size))
    
    # Normalize to 0-1 range
    image = image.astype(np.float32) / 255.0
    
    # Reshape for model input (add channel dimension)
    return image.reshape(1, target_size, target_size)

def normalize_image(image):
    """Normalizes image to 0-255 range."""
    return ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)


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
    """
    Pads image to square dimensions with background color.
    
    Args:
        image (np.ndarray): Input image
        size (int, optional): Target size. Defaults to 28.
    
    Returns:
        np.ndarray: Padded square image
    """
    h, w = image.shape
    max_side = max(h, w)
    
    # Use the minimum value from the image as padding
    pad_value = np.max(image)
    padded = np.full((max_side, max_side), pad_value * 0.85, dtype=np.uint8)
    
    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return padded

