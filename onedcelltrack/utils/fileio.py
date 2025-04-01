"""
fileio.py
=========
Utility functions for file input/output operations.

This module provides helper functions for:
  - Loading ND2 and TIFF image files.
  - Saving and loading NumPy arrays.
  - Reading and writing JSON configuration files.
  - Listing files in a directory matching a given pattern.

Adjust and extend these functions as needed for your experiments.
"""

import os
import json
import numpy as np
from glob import glob
from tifffile import imread, imwrite
from nd2reader import ND2Reader

def load_nd2(filepath: str, frames: tuple[int, int] | None = None, channel: int = 0, fov: int = 0) -> np.ndarray:
    """
    Load an ND2 file and return a NumPy array containing the specified frames.

    Parameters:
        filepath (str): Path to the ND2 file.
        frames (tuple[int, int] | None): Optional tuple (start, end) to select a range of frames.
                                         If None, all frames are loaded.
        channel (int): Index of the channel to load (default: 0).
        fov (int): Field-of-view index (default: 0).

    Returns:
        np.ndarray: Array containing the selected frames.
    """
    reader = ND2Reader(filepath)
    if frames is not None:
        start, end = frames
        frame_indices = list(range(start, end))
    else:
        frame_indices = list(range(reader.sizes['t']))
    # Retrieve the requested frames for the given channel and FOV.
    data = np.array([reader.get_frame_2D(t=i, c=channel, v=fov) for i in frame_indices])
    return data

def load_tiff(filepath: str) -> np.ndarray:
    """
    Load a TIFF file and return the image data as a NumPy array.

    Parameters:
        filepath (str): Path to the TIFF file.

    Returns:
        np.ndarray: Loaded image data.
    """
    return imread(filepath)

def save_npy(array: np.ndarray, filepath: str) -> None:
    """
    Save a NumPy array to a .npy file.

    Parameters:
        array (np.ndarray): Array to be saved.
        filepath (str): Path to the output file.
    """
    np.save(filepath, array)

def load_npy(filepath: str) -> np.ndarray:
    """
    Load a NumPy array from a .npy file.

    Parameters:
        filepath (str): Path to the .npy file.

    Returns:
        np.ndarray: Loaded NumPy array.
    """
    return np.load(filepath)

def read_json(filepath: str) -> dict:
    """
    Read a JSON file and return its contents as a dictionary.

    Parameters:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def write_json(data: dict, filepath: str, indent: int = 2) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters:
        data (dict): Data to write.
        filepath (str): Path to the output JSON file.
        indent (int): Indentation level for pretty-printing (default: 2).
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)

def list_files(directory: str, pattern: str = "*") -> list[str]:
    """
    List all files in a directory matching a given glob pattern.

    Parameters:
        directory (str): Directory to search.
        pattern (str): Glob pattern to match files (default "*").

    Returns:
        list[str]: List of matching file paths.
    """
    return glob(os.path.join(directory, pattern))

if __name__ == "__main__":
    # Simple test routine for standalone testing.
    import sys
    print("Testing fileio module...")
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        ext = os.path.splitext(filepath)[-1].lower()
        if ext == ".nd2":
            print(f"Loading ND2 file: {filepath}")
            data = load_nd2(filepath)
            print(f"Loaded data shape: {data.shape}")
        elif ext in [".tif", ".tiff"]:
            print(f"Loading TIFF file: {filepath}")
            data = load_tiff(filepath)
            print(f"Loaded data shape: {data.shape}")
        else:
            print("Unsupported file type.")
    else:
        print("No file provided for testing.")
