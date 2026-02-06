import os
import torch
import pickle as pkl
import numpy as np
import torchvision.transforms as transforms
from typing import Any
import json
import PIL.Image as Image
import random

CROWD_DATA_TRANSFORM = transforms.Compose([
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


def load_nwpu_ground_truth_points_from_path(ground_truth_path: str) -> np.ndarray:
    filetype = os.path.basename(ground_truth_path).split(".")[-1].lower()

    if filetype == "json":
        with open(ground_truth_path, "r") as f:
            data = json.load(f)
            gt_points = data["points"]
            return np.array(gt_points)
    elif filetype == "mat":
        # TODO Implement for .mat files
        pass



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


def load_nwpu_data_from_path(image_path: str, ground_truth_path: str) -> tuple[str, np.ndarray]:
    if not os.path.exists(image_path):
        raise ValueError(f"Image path not found: {image_path}")
        
    if not os.path.exists(ground_truth_path):
        raise ValueError(f"Ground Truth path not found: {ground_truth_path}")
        
    if not (os.path.basename(image_path).split(".")[0] == os.path.basename(ground_truth_path).split(".")[0]):
        raise ValueError(f"Image path: {image_path} does not match to ground truth path: {ground_truth_path}")

    ground_truth_points = load_nwpu_ground_truth_points_from_path(ground_truth_path)
    return image_path, ground_truth_points
    
    

def split_crowd_data_into_density_bins(image_gt_pairs: list[tuple[str, np.ndarray]]) -> dict[str, list[tuple[str, np.ndarray]]]:
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


def crowd_collate_fn(batch: list[tuple[torch.Tensor, Any]]) -> tuple[torch.Tensor, list[Any]]:
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


def center_crop_image(img: Image.Image, ground_truth_points: np.ndarray, 
                      crop_size: int=128) -> tuple[Image.Image, np.ndarray]:
    '''
    Center crops the image to the specified crop size and adjusts the ground truth points accordingly.
    If the image is smaller than the crop size, it is first padded to the crop size.

    Args:
        img (PIL.Image.Image): The input image to be cropped.
        ground_truth_points (np.ndarray): Numpy array of shape (N, 2) containing ground truth points.
        crop_size (int, optional): The size of the crop. Defaults to 128.
    
    Returns:
        (tuple[PIL.Image.Image, np.ndarray]): A tuple containing the cropped image and the adjusted ground truth points.
    '''
    img, ground_truth_points = pad_image_for_crop(img, ground_truth_points, crop_size)
    width, height = img.size

    left = (width - crop_size) // 2
    upper = (height - crop_size) // 2
    right = left + crop_size
    lower = upper + crop_size
    img = img.crop((left, upper, right, lower))
        
    gt_points_cropped = crop_ground_truth_points((left, upper, right, lower), ground_truth_points)

    return img, gt_points_cropped
    

def random_crop_image(img: Image.Image, ground_truth_points: np.ndarray, 
                      crop_size: int=128) -> tuple[Image.Image, np.ndarray]:
    '''
    Randomly crops the image to the specified crop size and adjusts the ground truth points accordingly.
    If the image is smaller than the crop size, it is first padded to the crop size.

    Args:
        img (PIL.Image.Image): The input image to be cropped.
        ground_truth_points (np.ndarray): Numpy array of shape (N, 2) containing ground truth points.
        crop_size (int, optional): The size of the crop. Defaults to 128.
    
    Returns:
        (tuple[PIL.Image.Image, np.ndarray]): A tuple containing the cropped image and the adjusted ground truth points.
    '''

    img, ground_truth_points = pad_image_for_crop(img, ground_truth_points, crop_size)
        
    width, height = img.size
    left = random.randint(0, width - crop_size)
    upper = random.randint(0, height - crop_size)
    right = left + crop_size
    lower = upper + crop_size
    img = img.crop((left, upper, right, lower))

    gt_points_cropped = crop_ground_truth_points((left, upper, right, lower), ground_truth_points)
        
    return img, gt_points_cropped


def resize_image_to_target_size(img: Image.Image, ground_truth_points: np.ndarray, 
                                target_size: tuple[int, int]) -> tuple[Image.Image, np.ndarray]:
    '''
    Resizes the image to the target size and adjusts the ground truth points accordingly.

    Args:
        img (PIL.Image.Image): The input image to be resized.
        ground_truth_points (np.ndarray): Numpy array of shape (N, 2) containing ground truth points.
        target_size (tuple[int, int]): The target size as (width, height).
    
    Returns:
        (tuple[PIL.Image.Image, np.ndarray]): A tuple containing the resized image and the adjusted ground truth points.
    '''
    width, height = img.size
    target_width, target_height = target_size
    factor_width, factor_height = target_width / width, target_height / height
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    ground_truth_points = np.array([[x * factor_width, y * factor_height] for x, y in ground_truth_points])
    return img, ground_truth_points


def resize_to_multiple_of_128(img: Image.Image, ground_truth_points: np.ndarray) \
    -> tuple[Image.Image, np.ndarray]:
    '''
    Resizes the image so that both its width and height are multiples of 128.
    Adjusts the ground truth points accordingly.

    Args:
        img (PIL.Image.Image): The input image to be resized.
        ground_truth_points (np.ndarray): Numpy array of shape (N, 2) containing ground truth points.
    
    Returns:
        (tuple[PIL.Image.Image, np.ndarray]): A tuple containing the resized image and the adjusted ground truth points.
    '''
    width, height = img.size
    # height or width already a multiple of 128?
    if width >= 128 and width // 128 == 0:
        new_width = width
    else: 
        new_width = max(128, (width // 128) * 128)
        
    if height >= 128 and height // 128 == 0:
        new_height = height
    else:
        new_height = max(128, (height // 128) * 128)
        
    if new_height == height and new_width == width:
        return img, ground_truth_points
        
    factor_width, factor_height = new_width / width, new_height / height
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    ground_truth_points_out = np.array([[x * factor_width, y * factor_height] for x, y in ground_truth_points])
    assert ground_truth_points_out.shape == ground_truth_points.shape
    return img, ground_truth_points_out
    

def crop_ground_truth_points(crop_box: tuple[int, int, int, int], 
                             ground_truth_points: np.ndarray) -> np.ndarray:
    '''
    Crops the ground truth points to fit within the specified crop box.

    Args:
        crop_box (tuple[int, int, int, int]): The crop box defined as (left, upper, right, lower).
        ground_truth_points (np.ndarray): Numpy array of shape (N, 2) containing ground truth points.
    
    Returns:
        np.ndarray: Numpy array of cropped ground truth points.
    '''
    left, upper, right, lower = crop_box
    if ground_truth_points.shape[0] > 0:
        mask = (ground_truth_points[:, 0] >= left) & (ground_truth_points[:, 0] < right) & \
            (ground_truth_points[:, 1] >= upper) & (ground_truth_points[:, 1] < lower)
        gt_points_cropped = ground_truth_points[mask].copy()
        gt_points_cropped[:, 0] -= left
        gt_points_cropped[:, 1] -= upper
    else:
        gt_points_cropped = np.zeros((0, 2))
        
    return gt_points_cropped
    

def scale_image_for_crop(img: Image.Image, ground_truth_points: np.ndarray, 
                              target_crop_size: int=128) -> tuple[Image.Image, np.ndarray]:
    '''
    Scales the image so that its smaller dimension is at least the target crop size.
    Adjusts the ground truth points accordingly.

    Args:
        img (PIL.Image.Image): The input image to be scaled.
        ground_truth_points (np.ndarray): Numpy array of shape (N, 2) containing ground truth points.
        target_crop_size (int, optional): The target crop size. Defaults to 128.
    
    Returns:
        (tuple[PIL.Image.Image, np.ndarray]): A tuple containing the scaled image and the adjusted ground truth points.
    '''
    width, height = img.size
    if width < target_crop_size or height < target_crop_size:
        scale = target_crop_size / min(width, height)
        new_width = int(np.ceil(width * scale))
        new_height = int(np.ceil(height * scale))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        ground_truth_points = ground_truth_points * scale

    return img, ground_truth_points
    

def pad_image_for_crop(img: Image.Image, ground_truth_points: np.ndarray,
                        target_crop_size: int=128) -> tuple[Image.Image, np.ndarray]:
    '''
    Pads the image so that both its width and height are at least the target crop size.
    The image is padded with black pixels (0,0,0).
    The image is centered in the padded image.
    Adjusts the ground truth points accordingly.

    Args:
        img (PIL.Image.Image): The input image to be padded.
        ground_truth_points (np.ndarray): Numpy array of shape (N, 2) containing ground truth points.
        target_crop_size (int, optional): The target crop size. Defaults to 128.
    
    Returns:
        (tuple[PIL.Image.Image, np.ndarray]): A tuple containing the padded image and the adjusted ground truth points.
    '''
    width, height = img.size
    if width < target_crop_size or height < target_crop_size:
        new_width = max(width, target_crop_size)
        new_height = max(height, target_crop_size)
        new_img = Image.new("RGB", (new_width, new_height), (0, 0, 0))

        pad_left = (new_width - width) // 2
        pad_top = (new_height - height) // 2
        new_img.paste(img, (pad_left, pad_top))
        img = new_img
        if ground_truth_points.shape[0] > 0:
            ground_truth_points_padded = ground_truth_points.copy()
            ground_truth_points_padded[:, 0] += pad_left
            ground_truth_points_padded[:, 1] += pad_top
            ground_truth_points = ground_truth_points_padded
        
    return img, ground_truth_points