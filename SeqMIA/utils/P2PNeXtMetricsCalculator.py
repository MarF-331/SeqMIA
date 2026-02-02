'''
This module provides utility functions to calculate various metrics for evaluating
the performance of the P2PNeXt model in head detection tasks.
'''

from enum import Enum
import numpy as np
import torch
from .P2PNext_utils import process_p2pnext_output
from .nap_and_ap_calculator import get_nap_score, get_ap_score
from functools import partial
from typing import Any, Optional


def ground_truth_count(target_points: np.ndarray) -> int:
    '''
    Extracts the ground truth people count by counting the points inside the provided target points.

    Args:
        target_points (np.ndarray): The pixel coordinates of the actual head positions. Shape [Num_Target_Points, 2]
    
    Returns:
        out (int): The actual number of people.
    '''
    if target_points.size == 0:
        return 0
    else:
        return target_points.shape[0]


def confidences_max(scores: np.ndarray) -> Any:
    '''
    Calculates the maximum value of all provided confidence scores.
    
    Args:
        scores (np.ndarray): Confidence scores. Shape [Num_Points, ]

    Returns:
        out: maximum score 
    '''
    if scores.size == 0:
        return 0.0
    else:
        return np.max(scores, axis=0)
    

def confidences_min(scores: np.ndarray) -> Any:
    '''
    Calculates the minimum value of all provided confidence scores.

    Args:
        scores (np.ndarray): Confidence scores. Shape [Num_Points, ]
    
    Returns:
        out: minimum score
    '''
    if scores.size == 0:
        return 0.0
    else:
        return np.min(scores, axis=0)


def confidences_mean(scores: np.ndarray) -> Any:
    '''
    Calculates the mean value of all provided confidence scores.

    Args:
        scores (np.ndarray): Confidence scores. Shape [Num_Points, ]
    
    Returns:
        out: mean score
    '''
    if scores.size == 0:
        return 0.0
    else:
        return np.mean(scores, axis=0)


def confidences_std(scores: np.ndarray) -> Any:
    '''
    Calculates the standard deviation of all provided confidence scores.

    Args:
        scores (np.ndarray): Confidence scores. Shape [Num_Points, ]
    
    Returns:
        out: standard deviation score
    
    '''
    if scores.size == 0:
        return 0.0
    else:
        return np.std(scores, axis=0)


def confidences_over_threshold(scores: np.ndarray, threshold: float=0.5) -> int:
    '''
    Calculates the number of confidence scores that are above a certain threshold.

    Args:
        scores (np.ndarray): Confidence scores. Shape [Num_Points, ]
        threshold (float, optional): The threshold to count scores above.
    
    Returns:
        out (int): number of scores above the threshold
    '''
    if scores.size == 0:
        return 0
    else:
        return int(np.sum(scores > threshold))


def count_error(predicted_points: np.ndarray, target_points: np.ndarray) -> int:
    '''
    Calculates the counting error of the prediction by comparing 
    the amount of predicted points to the amount of target points.

    Args:
        predicted_points (np.ndarray): The pixel coordinates of the predicted head positions. Shape [Num_Pred_Points, 2]
        target_points (np.ndarray): The pixel coordinates of the actual head positions. Shape [Num_Target_Points, 2]
        
    Returns:
        out (int): counting error abs(predicted count - target count)
    '''
    pred_count = predicted_points.shape[0]
    target_count = target_points.shape[0]
    return abs(pred_count - target_count)


def nap(predicted_points: np.ndarray, predicted_confidence_scores: np.ndarray, 
        target_points: np.ndarray, k: int=4, threshold: float=0.5) -> float:
    '''
    Calculates the normalized Average Precision (nAP) score.

    Args:
        predicted_points (np.ndarray): The predicted head positions from P2PNeXt. Shape [Num_Pred_Points, 2]
        predicted_confidence_scores (np.ndarray): The confidence scores to the predicted head positions. Shape [Num_Pred_Points, ]
        target_points (np.ndarray): The ground truth head positions. Shape [Num_Target_Points, 2]
        k (int, optional): The k value for k-nearest neighbors distance calculation.
        threshold (float, optional): The distance threshold multiplier for a correct detection.
    
    Returns:
        out (float): nAP score
    '''
    nap_score, _, _ = get_nap_score(predicted_points, predicted_confidence_scores, 
                                    target_points, k, threshold)
    return nap_score


def ap(predicted_points: np.ndarray, predicted_confidence_scores: np.ndarray,
       target_points: np.ndarray, threshold: int=8) -> float:
    '''
    Calculates the Average Precision (AP) score.

    Args:
        predicted_points (np.ndarray): The predicted head positions from P2PNeXt. Shape [Num_Pred_Points, 2]
        predicted_confidence_scores (np.ndarray): The confidence scores to the predicted head positions. Shape [Num_Pred_Points, ]
        target_points (np.ndarray): The ground truth head positions. Shape [Num_Target_Points, 2]
        threshold (int, optional): The distance threshold in pixels for a correct detection.
    
    Returns:
        out (float): AP score
    '''
    
    ap_score, _ = get_ap_score(predicted_points, predicted_confidence_scores, target_points, threshold)

    return ap_score


class MetricTypes(Enum):
    '''
    This enum defines the available metric types for P2PNeXt evaluation.
    Each metric type is associated with a function that computes the metric 
    and a string name for the metric.
    '''
    MeanConfidence = confidences_mean, confidences_mean.__name__
    StandardDerivationConfidence = confidences_std, confidences_std.__name__
    MaxConfidence = confidences_max, confidences_max.__name__
    MinConfidence = confidences_min, confidences_min.__name__
    ConfidencesOver99 = partial(confidences_over_threshold, threshold=0.99), "confidences_over_0.99"
    ConfidencesOver95 = partial(confidences_over_threshold, threshold=0.95), "confidences_over_0.95"
    ConfidencesOver90 = partial(confidences_over_threshold, threshold=0.90), "confidences_over_0.90"
    GroundTruthCount = ground_truth_count, ground_truth_count.__name__
    CountError = count_error, count_error.__name__
    NAP_K3_T001 = partial(nap, k=3, threshold=0.01), "nAP(k=3, d=0.01)"
    NAP_K3_T005 = partial(nap, k=3, threshold=0.05), "nAP(k=3, d=0.05)"
    NAP_K3_T01 = partial(nap, k=3, threshold=0.1), "nAP(k=3, d=0.1)"
    NAP_K3_T05 = partial(nap, k=3, threshold= 0.5), "nAP(k=3, d=0.5)"
    AP_16 = partial(ap, threshold=16), "AP(d=16)"
    AP_8 = partial(ap, threshold=8), "AP(d=8)"
    AP_4 = partial(ap, threshold=4), "AP(d=4)"
    AP_2 = partial(ap, threshold=2), "AP(d=2)"



def calculate(output_raw: dict[str, torch.Tensor], target: list[dict[str, torch.Tensor]], 
              metric_types: list[MetricTypes], model_id: Optional[str]=None) -> dict[dict[str, Any]]:
    '''
    Calculate specified metrics based on the model output and target data.
    A list of MetricTypes must be provided to specify which metrics to calculate.

    Args:
        output_raw (dict[str, torch.Tensor]): The raw output from the P2PNeXt model.
        target (list[dict[str, torch.Tensor]]): The ground truth target data.
        metric_types (list[MetricTypes]): A list of MetricTypes to calculate.
        model_id (Optional[str]): Optionally set a model id string to give each metric name in the output that model id as a suffix. This is useful when you have images processed by different models and want to distinguish the metrics in the output. 
    
    Returns:
        out (dict[dict[str, Any]]): A dictionary of dicionaries containing calculated metrics for each item in the batch.
        The outer dictionary takes image ids as keys. In case of JHU an image id would be something like "0432" or "0031".
        In the inner dictionary, each metric can be accessed with the name of the metric as the key.
    '''
    result: dict[dict[str, Any]] = {}
    processed_output = process_p2pnext_output(output_raw)
    batch_size = len(processed_output)

    for batch_idx in range(batch_size):
        current_pred_points, current_pred_scores = processed_output[batch_idx]
        current_target_points = target[batch_idx]["point"].cpu().numpy()
        current_image_id = target[batch_idx]["image_id"].cpu().item()
        current_image_metrics: dict = {}
        for metric_type in metric_types:
            match metric_type:
                case MetricTypes.MeanConfidence as mean_conf:
                    current_image_metrics[mean_conf.value[1]] = mean_conf.value[0](current_pred_scores)
                
                case MetricTypes.MaxConfidence as max_conf:
                    current_image_metrics[max_conf.value[1]] = max_conf.value[0](current_pred_scores)
                
                case MetricTypes.MinConfidence as min_conf:
                    current_image_metrics[min_conf.value[1]] = min_conf.value[0](current_pred_scores)
                
                case MetricTypes.StandardDerivationConfidence as std_conf:
                    current_image_metrics[std_conf.value[1]] = std_conf.value[0](current_pred_scores)
                
                case MetricTypes.ConfidencesOver99 as conf_over_99:
                    current_image_metrics[conf_over_99.value[1]] = conf_over_99.value[0](scores=current_pred_scores)
                
                case MetricTypes.ConfidencesOver95 as conf_over_95:
                    current_image_metrics[conf_over_95.value[1]] = conf_over_95.value[0](scores=current_pred_scores)

                case MetricTypes.ConfidencesOver90 as conf_over_90:
                    current_image_metrics[conf_over_90.value[1]] = conf_over_90.value[0](scores=current_pred_scores)
                
                case MetricTypes.GroundTruthCount as gt_count:
                    current_image_metrics[gt_count.value[1]] = gt_count.value[0](current_target_points)

                case MetricTypes.CountError as c_err:
                    current_image_metrics[c_err.value[1]] = \
                        c_err.value[0](predicted_points=current_pred_points, 
                                       target_points=current_target_points)
                
                case MetricTypes.NAP_K3_T001 as nap_k3_t001:
                    current_image_metrics[nap_k3_t001.value[1]] = \
                        nap_k3_t001.value[0](predicted_points=current_pred_points, 
                                             predicted_confidence_scores=current_pred_scores,
                                             target_points=current_target_points)
                
                case MetricTypes.NAP_K3_T005 as nap_k3_t005:
                    current_image_metrics[nap_k3_t005.value[1]] = \
                        nap_k3_t005.value[0](predicted_points=current_pred_points,
                                             predicted_confidence_scores=current_pred_scores,
                                             target_points=current_target_points)
                
                case MetricTypes.NAP_K3_T05 as nap_k3_t05:
                    current_image_metrics[nap_k3_t05.value[1]] = \
                        nap_k3_t05.value[0](predicted_points=current_pred_points, 
                                            predicted_confidence_scores=current_pred_scores,
                                            target_points=current_target_points)
                
                case MetricTypes.NAP_K3_T01 as nap_k3_t01:
                    current_image_metrics[nap_k3_t01.value[1]] = \
                        nap_k3_t01.value[0](predicted_points=current_pred_points,
                                            predicted_confidence_scores=current_pred_scores,
                                            target_points=current_target_points)
                
                case MetricTypes.AP_16 as ap_16:
                    current_image_metrics[ap_16.value[1]] = \
                        ap_16.value[0](predicted_points=current_pred_points,
                                       predicted_confidence_scores=current_pred_scores,
                                       target_points=current_target_points)
                
                case MetricTypes.AP_8 as ap_8:
                    current_image_metrics[ap_8.value[1]] = \
                        ap_8.value[0](predicted_points=current_pred_points,
                                      predicted_confidence_scores=current_pred_scores,
                                      target_points=current_target_points)
                
                case MetricTypes.AP_4 as ap_4:
                    current_image_metrics[ap_4.value[1]] = \
                        ap_4.value[0](predicted_points=current_pred_points,
                                      predicted_confidence_scores=current_pred_scores,
                                      target_points=current_target_points)
                
                case MetricTypes.AP_2 as ap_2:
                    current_image_metrics[ap_2.value[1]] = \
                        ap_2.value[0](predicted_points=current_pred_points,
                                      predicted_confidence_scores=current_pred_scores,
                                      target_points=current_target_points)
            
         # Append model id suffix if provided
        if model_id is not None:
            current_image_metrics = {f"{key}_{model_id}": value 
                                     for key, value in current_image_metrics.items()}
        
        result[current_image_id] = current_image_metrics

    return result

