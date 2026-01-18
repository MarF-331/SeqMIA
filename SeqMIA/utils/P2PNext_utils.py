import torch
import numpy as np

def process_p2pnext_output(output: dict[str, torch.Tensor], filter_threshold: float=0.5) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Process the raw output from the P2PNeXt model to extract head positions
    and filter them based on a confidence threshold.

    Args:
        output (dict[str,torch.Tensor]): Raw output from the P2PNeXt model.
        filter_threshold (float): Confidence threshold for filtering head positions.

    Returns:
        list[tuple[np.ndarray,np.ndarray]]: A list of tuples, each containing filtered head positions and their scores for each image in the batch.

    """
    outputs_scores = torch.nn.functional.softmax(output['pred_logits'], -1)[:, :, 1]
    outputs_points = output['pred_points']
    batch_size = outputs_points.shape[0]

    result = []
    for i in range(batch_size):
        scores = outputs_scores[i]
        points = outputs_points[i]

        filtered_points = points[scores > filter_threshold].detach().cpu().numpy()
        filtered_scores = scores[scores > filter_threshold].detach().cpu().numpy()
        
        result.append((filtered_points, filtered_scores))
    
    return result