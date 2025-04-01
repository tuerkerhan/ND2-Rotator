#!/usr/bin/env python3
"""
analysis.py
===========
Module for analyzing cell tracking data.

This module provides utilities for joining tracking data from multiple experiments,
augmenting the combined DataFrame with additional metrics (like unique IDs and durations),
and performing basic analyses on the trajectories.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Union

def join_dataframes(paths: Union[str, List[str]]) -> pd.DataFrame:
    """
    Join CSV tracking data from one or more experiments into a single DataFrame.

    Parameters:
        paths (str or List[str]): A directory or list of directories containing subdirectories 
                                  (with 'XY' in their name) that hold 'clean_tracking_data.csv' files.

    Returns:
        pd.DataFrame: Combined DataFrame with added 'fov' and 'experiment' columns.
    """
    if isinstance(paths, str):
        paths = [paths]

    dfs = []
    for path in paths:
        # Find subdirectories that likely represent fields-of-view (fov)
        fov_dirs = [d for d in os.listdir(path) if 'XY' in d]
        for fov_dir in tqdm(fov_dirs, desc=f"Processing FOV directories in {path}"):
            try:
                fov_value = float(fov_dir.split("XY")[-1])
            except ValueError:
                continue
            csv_path = os.path.join(path, fov_dir, "clean_tracking_data.csv")
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path, low_memory=False)
            df["fov"] = fov_value
            df["experiment"] = os.path.basename(path)
            dfs.append(df)
    if not dfs:
        raise ValueError("No valid tracking CSVs found in the provided paths.")
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def create_full_dataframe(paths: Union[str, List[str]], fpm: float = 0.5) -> pd.DataFrame:
    """
    Augment the combined DataFrame with a unique track identifier and compute the duration
    of each track based on frame information.

    Parameters:
        paths (str or List[str]): Path(s) to directories containing tracking data.
        fpm (float): Frame rate conversion factor (frames per minute). Default is 0.5.

    Returns:
        pd.DataFrame: Augmented DataFrame with 'unique_id' and 'duration' columns.
    """
    df = join_dataframes(paths)
    df = df.sort_values(by=["experiment", "fov", "particle", "segment", "frame"]).reset_index(drop=True)
    df["unique_id"] = 0
    df["duration"] = 0.0
    unique_id_counter = 0

    experiments = df["experiment"].unique()
    for exp in experiments:
        exp_df = df[df["experiment"] == exp]
        fovs = exp_df["fov"].unique()
        for fov in tqdm(fovs, desc=f"Processing experiment {exp}"):
            # Process each particle in this field-of-view
            particles = exp_df[(exp_df["experiment"] == exp) & (exp_df["fov"] == fov)]["particle"].unique()
            for particle in particles:
                particle_mask = (df["experiment"] == exp) & (df["fov"] == fov) & (df["particle"] == particle)
                segments = df.loc[particle_mask, "segment"].unique()
                segments = [seg for seg in segments if seg > 0]
                for segment in segments:
                    seg_mask = particle_mask & (df["segment"] == segment)
                    df.loc[seg_mask, "unique_id"] = unique_id_counter
                    frames = df.loc[seg_mask, "frame"]
                    duration = (frames.max() - frames.min()) / fpm
                    df.loc[seg_mask, "duration"] = duration
                    unique_id_counter += 1
    return df

class CellTrackAnalysis:
    """
    Class for analyzing cell tracking data.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing cell tracking information.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def sample_trajectories(self, n_samples: int) -> pd.DataFrame:
        """
        Randomly sample a subset of trajectories from the tracking data.
        
        Parameters:
            n_samples (int): Number of trajectories to sample.
        
        Returns:
            pd.DataFrame: DataFrame containing only the sampled trajectories.
        """
        # Sample based on unique particle IDs
        unique_particles = self.df["particle"].unique()
        n = min(n_samples, len(unique_particles))
        sampled_particles = np.random.choice(unique_particles, size=n, replace=False)
        return self.df[self.df["particle"].isin(sampled_particles)]

    def get_statistics(self) -> dict:
        """
        Compute basic statistics on the tracking data, such as duration metrics.
        
        Returns:
            dict: Dictionary containing mean, median, min, max durations and total track count.
        """
        stats = {
            "mean_duration": self.df["duration"].mean(),
            "median_duration": self.df["duration"].median(),
            "max_duration": self.df["duration"].max(),
            "min_duration": self.df["duration"].min(),
            "total_tracks": self.df["particle"].nunique(),
        }
        return stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze cell tracking data.")
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to directory containing experiment folders with tracking CSVs."
    )
    parser.add_argument(
        "--fpm", type=float, default=0.5,
        help="Frames per minute conversion factor (default: 0.5)."
    )
    args = parser.parse_args()

    try:
        df_full = create_full_dataframe(args.data_path, fpm=args.fpm)
        print("Combined DataFrame shape:", df_full.shape)
        analysis = CellTrackAnalysis(df_full)
        stats = analysis.get_statistics()
        print("Tracking statistics:", stats)
    except Exception as e:
        print(f"Error during analysis: {e}")
