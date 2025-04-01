"""
preprocessing.py
================
Module for loading and preprocessing images for the cell tracking pipeline.
This module is designed for Python 3.11 and works within the new Pixi-based project.
It currently supports loading TIFF images. Support for other formats (e.g. ND2) can be added as needed.
"""

import os
import logging
import numpy as np
from tifffile import imread

logger = logging.getLogger(__name__)

def load_images(cyto_path: str, nucleus_path: str, frame_indices: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load cytoplasm and nucleus images from file paths.

    Parameters:
        cyto_path (str): Path to the cytoplasm image file (TIFF recommended).
        nucleus_path (str): Path to the nucleus image file (TIFF recommended).
        frame_indices (list[int], optional): A two-element list [start, end] defining the frame range to load.
            If None, all frames will be loaded.

    Returns:
        tuple[np.ndarray, np.ndarray]: The cytoplasm and nucleus images as NumPy arrays.
    """
    # Determine file extensions to decide on loading method.
    ext_cyto = os.path.splitext(cyto_path)[-1].lower()
    ext_nucleus = os.path.splitext(nucleus_path)[-1].lower()

    if ext_cyto in [".tif", ".tiff"]:
        cyto_img = imread(cyto_path)
    else:
        raise NotImplementedError(f"File format '{ext_cyto}' for cytoplasm is not yet supported.")

    if ext_nucleus in [".tif", ".tiff"]:
        nucleus_img = imread(nucleus_path)
    else:
        raise NotImplementedError(f"File format '{ext_nucleus}' for nucleus is not yet supported.")

    # If frame indices are provided, slice the arrays accordingly.
    if frame_indices is not None:
        if len(frame_indices) != 2:
            raise ValueError("frame_indices must be a list or tuple with two elements: [start, end].")
        start, end = frame_indices
        cyto_img = cyto_img[start:end]
        nucleus_img = nucleus_img[start:end]

    logger.info("Loaded cytoplasm image of shape %s and nucleus image of shape %s", cyto_img.shape, nucleus_img.shape)
    return cyto_img, nucleus_img

def preprocess_image(
    image: np.ndarray,
    contrast: float = 1.0,
    brightness: float = 0.0,
    bottom_percentile: float = 0.5,
    top_percentile: float = 99.5
) -> np.ndarray:
    """
    Preprocess an image by clipping its intensities based on provided percentiles,
    normalizing to the [0, 1] range, and applying contrast and brightness adjustments.

    Parameters:
        image (np.ndarray): Input image array.
        contrast (float, optional): Multiplicative factor to adjust contrast. Defaults to 1.0.
        brightness (float, optional): Additive term to adjust brightness. Defaults to 0.0.
        bottom_percentile (float, optional): Lower percentile for intensity clipping. Defaults to 0.5.
        top_percentile (float, optional): Upper percentile for intensity clipping. Defaults to 99.5.

    Returns:
        np.ndarray: Preprocessed image with intensities scaled to [0, 1].
    """
    # Convert image to float32 for precise arithmetic
    image = image.astype(np.float32)

    # Compute intensity thresholds based on the given percentiles
    p_low = np.percentile(image, bottom_percentile)
    p_high = np.percentile(image, top_percentile)
    if p_high <= p_low:
        logger.warning("Percentile values result in zero range; skipping normalization.")
        norm_image = image
    else:
        # Clip the image and then normalize to [0, 1]
        norm_image = np.clip(image, p_low, p_high)
        norm_image = (norm_image - p_low) / (p_high - p_low)

    # Apply contrast and brightness adjustments
    adjusted_image = contrast * norm_image + brightness
    # Ensure values are within the valid [0, 1] range
    adjusted_image = np.clip(adjusted_image, 0.0, 1.0)

    return adjusted_image

# Optionally, include a main function for standalone testing
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 3:
        print("Usage: python preprocessing.py <cyto_path> <nucleus_path> [start:end]")
        sys.exit(1)
    cyto_path = sys.argv[1]
    nucleus_path = sys.argv[2]
    frame_indices = None
    if len(sys.argv) == 4:
        try:
            start, end = map(int, sys.argv[3].split(':'))
            frame_indices = [start, end]
        except Exception as e:
            print("Frame indices must be in format start:end")
            sys.exit(1)
    cyto_img, nuc_img = load_images(cyto_path, nucleus_path, frame_indices)
    cyto_pre = preprocess_image(cyto_img)
    nuc_pre = preprocess_image(nuc_img)
    print("Preprocessing complete.")
