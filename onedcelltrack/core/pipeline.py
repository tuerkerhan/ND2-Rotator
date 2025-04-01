#!/usr/bin/env python3
"""
Pipeline for cell tracking experiments.

This script orchestrates image loading, preprocessing, segmentation, and tracking.
It’s written for Python 3.11 and assumes the new project structure using Pixi.
Use it as a starting point and tweak it to fit your experiment’s needs.
"""

import argparse
import logging
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Import the refactored core modules.
from onedcelltrack.core import preprocessing, segmentation, tracking

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cell tracking pipeline: preprocess images, segment, and track cells."
    )
    parser.add_argument(
        "--cyto", type=str, required=True,
        help="Path to the cytoplasm image file (e.g. a TIFF or ND2 file)."
    )
    parser.add_argument(
        "--nucleus", type=str, required=True,
        help="Path to the nucleus image file (e.g. a TIFF or ND2 file)."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Directory to store output files."
    )
    parser.add_argument(
        "--frames", type=str, default=None,
        help="Frame range to process, e.g. '0-100'. Process all frames if not specified."
    )
    parser.add_argument(
        "--diameter", type=float, default=29.0,
        help="Estimated cell diameter used for segmentation and tracking."
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Enable GPU acceleration (if available) for segmentation."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging output."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )
    logger = logging.getLogger("cell_tracking_pipeline")
    logger.info("Starting cell tracking pipeline")
    
    # Ensure the output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine frame indices to process
    if args.frames:
        try:
            start, end = map(int, args.frames.split('-'))
            frame_indices = list(range(start, end))
        except Exception as e:
            logger.error("Invalid frame range format. Expected format 'start-end'.")
            raise e
    else:
        frame_indices = None  # Process all frames
    
    # 1. Preprocessing: load and preprocess images
    logger.info("Loading and preprocessing images")
    try:
        # load_images should be a function that returns cytoplasm and nucleus arrays.
        cyto_img, nuc_img = preprocessing.load_images(args.cyto, args.nucleus, frame_indices)
        
        # Apply preprocessing – adjust contrast/brightness as needed.
        cyto_pre = preprocessing.preprocess_image(cyto_img, contrast=1.0, brightness=0.0)
        nuc_pre = preprocessing.preprocess_image(nuc_img, contrast=1.0, brightness=0.0)
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise e
    
    # 2. Segmentation: run segmentation on the preprocessed images
    logger.info("Running segmentation")
    try:
        # The segmentation module should output masks as a NumPy array.
        masks = segmentation.segment(cyto_pre, nuc_pre, gpu=args.gpu, diameter=args.diameter)
    except Exception as e:
        logger.error("Segmentation failed: %s", e)
        raise e

    # 3. Tracking: track nuclei across frames
    logger.info("Performing tracking")
    try:
        # The tracking module should output a DataFrame of tracked particles.
        tracks: pd.DataFrame = tracking.track(nuc_pre, diameter=args.diameter)
    except Exception as e:
        logger.error("Tracking failed: %s", e)
        raise e
    
    # 4. Save outputs
    masks_file = output_dir / "masks.npy"
    tracks_file = output_dir / "tracks.csv"
    try:
        np.save(masks_file, masks)
        tracks.to_csv(tracks_file, index=False)
    except Exception as e:
        logger.error("Failed to save outputs: %s", e)
        raise e

    # Write a metadata log
    meta = {
        "cyto_file": args.cyto,
        "nucleus_file": args.nucleus,
        "frames": frame_indices,
        "diameter": args.diameter,
        "gpu": args.gpu,
    }
    meta_file = output_dir / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info("Pipeline complete. Masks saved to '%s' and tracking data saved to '%s'.", masks_file, tracks_file)

if __name__ == "__main__":
    main()
