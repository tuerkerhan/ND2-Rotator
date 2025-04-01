"""
tracking.py
===========
Module for tracking nuclei positions using trackpy.

This module implements a function 'track' that detects nuclei in a preprocessed
nuclei image stack and links these detections into trajectories using trackpy.
It returns a pandas DataFrame with the linked tracks.
"""

import numpy as np
import trackpy as tp
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def track(
    nuclei: np.ndarray,
    diameter: float = 19,
    minmass: float | None = None,
    track_memory: int = 15,
    max_travel: float = 5,
    min_frames: int = 10,
    pixel_to_um: float = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Detect nuclei positions in a fluorescence image stack and link them into trajectories.

    Parameters:
        nuclei (np.ndarray): 3D numpy array representing preprocessed nuclei images (frames, height, width).
        diameter (float, optional): Estimated diameter of nuclei. Default is 19.
        minmass (float, optional): Minimum integrated brightness for a detection.
            If None, a default is chosen based on the image's data type.
        track_memory (int, optional): Maximum number of frames a nucleus can be missing and still be linked.
            Default is 15.
        max_travel (float, optional): Maximum distance (in pixels) a nucleus can travel between frames.
            Default is 5.
        min_frames (int, optional): Minimum number of frames a track must span to be considered valid.
            Default is 10.
        pixel_to_um (float, optional): Conversion factor from pixels to micrometers (currently not used).
            Default is 1.
        verbose (bool, optional): If True, enables verbose tracking output.

    Returns:
        pd.DataFrame: DataFrame containing linked trajectories. Columns include frame, x, y, particle, etc.
    """
    # Ensure the estimated diameter is odd.
    if diameter % 2 == 0:
        diameter += 1

    # Set a default minmass based on image type if not provided.
    if minmass is None:
        if nuclei.dtype == 'uint8':
            minmass = 3500
        elif nuclei.dtype == 'uint16':
            minmass = 2500 * (2**16) / 255
        else:
            minmass = 3500  # Fallback default

    if not verbose:
        tp.quiet()

    logger.info("Starting detection with trackpy: diameter=%s, minmass=%s", diameter, minmass)
    # Detect features (nuclei) in all frames.
    features = tp.batch(nuclei, diameter=diameter, minmass=minmass)
    logger.info("Detection complete. Found %d features.", len(features))

    # Link detected features into trajectories.
    max_travel = np.round(max_travel)
    logger.info("Linking features: max_travel=%s, memory=%s", max_travel, track_memory)
    trajectories = tp.link(features, max_travel, memory=track_memory)
    
    # Filter out trajectories that are too short.
    trajectories = tp.filter_stubs(trajectories, min_frames)
    unique_particles = trajectories['particle'].unique()
    logger.info("Tracking complete. %d valid tracks remaining.", len(unique_particles))

    return trajectories

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Configure basic logging for standalone testing.
    logging.basicConfig(level=logging.INFO)

    # Create a dummy nuclei image stack for testing (10 frames, 256x256).
    n_frames, height, width = 10, 256, 256
    dummy_nuclei = (np.random.rand(n_frames, height, width) * 255).astype("uint8")
    
    # Run tracking.
    tracks_df = track(dummy_nuclei, verbose=True)
    print(tracks_df.head())
    
    # Visualize detections on the first frame.
    frame_number = 0
    frame = dummy_nuclei[frame_number]
    features = tracks_df[tracks_df['frame'] == frame_number]
    
    plt.imshow(frame, cmap="gray")
    plt.scatter(features['x'], features['y'], edgecolors="red", facecolors="none")
    plt.title("Detected Nuclei in Frame 0")
    plt.show()
