"""
segmentation.py
===============
Module for segmenting cells using Cellpose.

This module implements segmentation functions for processing preprocessed
cytoplasm and nucleus images. It supports both batch segmentation and looped
(frame-by-frame) segmentation. It also provides an internal helper to load a
pretrained model if one is specified.
"""

import os
import json
import numpy as np
from urllib import request
from tqdm import tqdm

try:
    from cellpose import models
except ImportError as e:
    raise ImportError("Cellpose is required for segmentation. Please install it via pip install cellpose") from e


def _load_pretrained_model(pretrained_model: str, gpu: bool, model_type: str) -> models.CellposeModel:
    """
    Internal helper to load a pretrained model.
    Looks up the model key in a local models.json file (located in a 'models' directory)
    and downloads the model if not present locally.
    
    Parameters:
        pretrained_model (str): Key name of the pretrained model.
        gpu (bool): Whether to use GPU acceleration.
        model_type (str): Model type for Cellpose (e.g., 'cyto').

    Returns:
        models.CellposeModel: An instance of the CellposeModel with the pretrained model.
    """
    path_to_models = os.path.join(os.path.dirname(__file__), "models")
    models_json = os.path.join(path_to_models, "models.json")
    with open(models_json, "r") as f:
        model_dict = json.load(f)
    if pretrained_model in model_dict:
        model_info = model_dict[pretrained_model]
        model_path = os.path.join(path_to_models, model_info["path"])
        if not os.path.isfile(model_path):
            print("Downloading model from remote...")
            url = model_info["link"]
            request.urlretrieve(url, model_path)
        return models.CellposeModel(gpu=gpu, pretrained_model=model_path)
    else:
        raise ValueError(f"Pretrained model '{pretrained_model}' not found in models.json.")


def segment(
    cytoplasm: np.ndarray,
    nucleus: np.ndarray,
    gpu: bool = True,
    model_type: str = "cyto",
    channels: list[int] | None = None,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    mask_threshold: float = 0,
    pretrained_model: str | None = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Segment cells using Cellpose based on cytoplasm and nucleus images.
    
    Parameters:
        cytoplasm (np.ndarray): Preprocessed cytoplasm image stack (frames, height, width).
        nucleus (np.ndarray): Preprocessed nucleus image stack (frames, height, width).
        gpu (bool): Whether to use GPU acceleration.
        model_type (str): Cellpose model type (default 'cyto').
        channels (list[int], optional): List defining channels. Defaults to [1, 2].
        diameter (float, optional): Estimated cell diameter.
        flow_threshold (float): Flow threshold for Cellpose (default 0.4).
        mask_threshold (float): Mask threshold for Cellpose (default 0).
        pretrained_model (str, optional): Key for a pretrained model defined in models.json.
        verbose (bool): Enable verbose output (default True).
    
    Returns:
        np.ndarray: Segmentation masks as a uint8 NumPy array.
    """
    if channels is None:
        channels = [1, 2]
    
    # Initialize Cellpose model
    if pretrained_model is None:
        model = models.Cellpose(gpu=gpu, model_type=model_type)
    else:
        model = _load_pretrained_model(pretrained_model, gpu, model_type)
    
    # Stack cytoplasm and nucleus images along a new channel axis.
    # Assumes input images have shape (n_frames, height, width) so that the output is (n_frames, 2, height, width)
    images = np.stack((cytoplasm, nucleus), axis=1)
    
    print("Running Cellpose segmentation on batch images...")
    masks = model.eval(
        images,
        diameter=diameter,
        channels=channels,
        flow_threshold=flow_threshold,
        mask_threshold=mask_threshold,
        normalize=True,
        verbose=verbose
    )[0].astype("uint8")
    
    return masks


def segment_looped(
    cytoplasm: np.ndarray,
    nucleus: np.ndarray,
    gpu: bool = True,
    model_type: str = "cyto",
    channels: list[int] | None = None,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    mask_threshold: float = 0,
    pretrained_model: str | None = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Segment cells using Cellpose in a loop over each frame.
    This approach is useful when memory is limited or for processing very large image stacks.
    
    Parameters:
        cytoplasm (np.ndarray): Preprocessed cytoplasm image stack (frames, height, width).
        nucleus (np.ndarray): Preprocessed nucleus image stack (frames, height, width).
        gpu (bool): Whether to use GPU acceleration.
        model_type (str): Cellpose model type (default 'cyto').
        channels (list[int], optional): List defining channels. Defaults to [1, 2].
        diameter (float, optional): Estimated cell diameter.
        flow_threshold (float): Flow threshold for Cellpose (default 0.4).
        mask_threshold (float): Mask threshold for Cellpose (default 0).
        pretrained_model (str, optional): Key for a pretrained model defined in models.json.
        verbose (bool): Enable verbose output (default True).
    
    Returns:
        np.ndarray: Segmentation masks for each frame as a uint8 NumPy array.
    """
    if channels is None:
        channels = [1, 2]
    
    # Initialize Cellpose model
    if pretrained_model is None:
        model = models.Cellpose(gpu=gpu, model_type=model_type)
    else:
        model = _load_pretrained_model(pretrained_model, gpu, model_type)
    
    n_frames = cytoplasm.shape[0]
    masks = np.zeros(cytoplasm.shape, dtype="uint8")
    
    print("Running Cellpose segmentation frame-by-frame...")
    for i in tqdm(range(n_frames), desc="Segmenting frames"):
        # Create a two-channel image for the current frame.
        image = np.stack((cytoplasm[i], nucleus[i]), axis=0)
        mask = model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            mask_threshold=mask_threshold,
            normalize=True,
            verbose=verbose
        )[0].astype("uint8")
        masks[i] = mask
    return masks


if __name__ == "__main__":
    # Standalone testing of segmentation functions with dummy data
    import matplotlib.pyplot as plt

    # Create dummy preprocessed images (5 frames, 256x256)
    dummy_cyto = np.random.rand(5, 256, 256).astype("float32")
    dummy_nuc = np.random.rand(5, 256, 256).astype("float32")
    
    # Test batch segmentation
    seg_masks = segment(dummy_cyto, dummy_nuc, gpu=False, verbose=True)
    print("Batch segmentation masks shape:", seg_masks.shape)
    plt.imshow(seg_masks[0], cmap="gray")
    plt.title("Segmentation Mask - Batch")
    plt.show()
    
    # Test looped segmentation
    seg_masks_loop = segment_looped(dummy_cyto, dummy_nuc, gpu=False, verbose=True)
    print("Looped segmentation masks shape:", seg_masks_loop.shape)
    plt.imshow(seg_masks_loop[0], cmap="gray")
    plt.title("Segmentation Mask - Looped")
    plt.show()
