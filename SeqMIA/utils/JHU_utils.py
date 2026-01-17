import os
import numpy as np

def load_ground_truth_points_from_path(ground_truth_path: str) -> np.ndarray:
    '''
    Returns the ground truth points as a numpy array given a path to file specifying the ground truth points like a .txt or a .mat file
    
    :param gt_path: Path to the file specifying the ground truth points.
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
    if not os.path.exists(image_path):
        raise ValueError(f"Image path not found: {image_path}")
    
    if not os.path.exists(ground_truth_path):
        raise ValueError(f"Ground Truth path not found: {ground_truth_path}")
    
    if not (os.path.basename(image_path).split(".")[0] == os.path.basename(ground_truth_path).split(".")[0]):
        raise ValueError(f"Image path: {image_path} does not match to ground truth path: {ground_truth_path}")
    
    ground_truth_points = load_ground_truth_points_from_path(ground_truth_path)
    return image_path, ground_truth_points