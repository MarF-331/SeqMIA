import os
import torch
import pickle as pkl
import numpy as np
import torchvision.transforms as transforms
from typing import Any

JHU_DATA_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_ground_truth_points_from_path(ground_truth_path: str) -> np.ndarray:
    '''
    Returns the ground truth points as a numpy array given a path to file specifying the ground truth points like a .txt or a .mat file
    
    Args:
        ground_truth_path (str): Path to the ground truth file.

    Returns:
        np.ndarray: Numpy array of shape (N, 2) where N is the number of ground truth points, and each point is represented by its (x, y) coordinates.
    '''
    # load ground truth points
    points = []

    with open(ground_truth_path, encoding='utf-8') as f_label:
        for line in f_label:
            line = line.strip().split()
            if len(line) >= 2:
                try:
                    x = float(line[0])
                    y = float(line[1])
                    points.append([x, y])
                except ValueError:
                    pass
    
    return np.array(points)


def load_jhu_data_from_path(image_path: str, ground_truth_path: str) -> tuple[str, np.ndarray]:
    '''
    Loads a tuple of image path and ground truth points from the specified paths.
    The image path and ground truth path must correspond to each other or else a ValueError is raised.
    
    Args:
        image_path (str): Path to the image file.
        ground_truth_path (str): Path to the ground truth file.
    
    Returns:
        tuple[str,np.ndarray]: A tuple containing the image path and the ground truth points as a numpy array.
    '''
    if not os.path.exists(image_path):
        raise ValueError(f"Image path not found: {image_path}")
    
    if not os.path.exists(ground_truth_path):
        raise ValueError(f"Ground Truth path not found: {ground_truth_path}")
    
    if not (os.path.basename(image_path).split(".")[0] == os.path.basename(ground_truth_path).split(".")[0]):
        raise ValueError(f"Image path: {image_path} does not match to ground truth path: {ground_truth_path}")
    
    ground_truth_points = load_ground_truth_points_from_path(ground_truth_path)
    return image_path, ground_truth_points


def split_jhu_data_into_density_bins(image_gt_pairs: list[tuple[str, np.ndarray]]) -> dict[str, list[tuple[str, np.ndarray]]]:
    '''
    Splits the JHU dataset into different density bins based on the number of ground truth points per image.
    The bins are defined as follows:
    - very_low: 0-35 points
    - low: 36-75 points
    - medium: 76-150 points
    - high: 151-500 points
    - very_high: 501-5000 points
    - super_high: 5001+ points
    
    Args:
        image_gt_pairs (list[tuple[str, np.ndarray]]): List of tuples containing image paths and their corresponding ground truth points.
    
    Returns:
        dict[str,list[tuple[str, np.ndarray]]]: Dictionary with keys as density bin names and values as lists of image-ground truth point tuples.
    '''
    density_bins = {
        "very_low": [],
        "low": [],
        "medium": [],
        "high": [],
        "very_high": [],
        "super_high": []
    }

    for image_path, gt_points in image_gt_pairs:
        num_points = gt_points.shape[0]
        if num_points <= 35:
            density_bins["very_low"].append((image_path, gt_points))
        elif num_points <= 75:
            density_bins["low"].append((image_path, gt_points))
        elif num_points <= 150:
            density_bins["medium"].append((image_path, gt_points))
        elif num_points <= 500:
            density_bins["high"].append((image_path, gt_points))
        elif num_points <= 5000:
            density_bins["very_high"].append((image_path, gt_points))
        else:
            density_bins["super_high"].append((image_path, gt_points))
    
    return density_bins


def save_split_to_pickle(save_path: str, **splits: list[tuple[str, np.ndarray]]) -> None:
    '''
    Saves the split information to a pickle file at the specified path.
    
    Args:
        save_path (str): Path to save the pickle file.
        **splits (list[tuple[str, np.ndarray]]): Keyword arguments where keys are split names and values are lists of image-ground truth point tuples.
    '''
    split_info = {key: value for key, value in splits.items()}
    with open(save_path, 'wb') as f:
        pkl.dump(split_info, f)
    print(f"Saved split information to {save_path}")


def load_split_from_pickle(load_path: str) -> dict[str, list[tuple[str, np.ndarray]]]:
    '''
    Loads the split information from a pickle file at the specified path.
    
    Args:
        load_path (str): Path to the pickle file.
    Returns:
        dict[str,list[tuple[str, np.ndarray]]]: Dictionary with keys as split names and values as lists of image-ground truth point tuples.
    '''
    with open(load_path, 'rb') as f:
        split_info = pkl.load(f)
    return split_info


def jhu_collate_fn(batch: list[tuple[torch.Tensor, Any]]) -> tuple[torch.Tensor, list[Any]]:
    '''
    A collate function for JHU dataset to stack image tensors and aggregate targets.
    Images must be of the same size before using this collate function.
    
    Args:
        batch (list[tuple[torch.Tensor, Any]]): List of tuples containing image tensors and their corresponding targets.
    
    Returns:
        tuple[torch.Tensor,list[Any]]: A tuple containing stacked image tensors and a list of targets.
    '''
    image_tensors_stacked = torch.stack([tensors for tensors, _ in batch])
    targets_stacked = [targets for _, targets in batch]
    return image_tensors_stacked, targets_stacked