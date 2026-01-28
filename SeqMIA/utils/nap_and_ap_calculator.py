import numpy as np
import logging
from scipy.spatial import KDTree
from sklearn.metrics import auc

logger = logging.getLogger(__name__)


def _sort_outputs_by_confidence_scores(predicted_points: np.ndarray, confidence_scores: np.ndarray) \
    -> tuple[np.ndarray, np.ndarray]:
    '''
    Sorts the outputs/predictions of the model in descending order by confidence scores.
    
    Args:
        predicted_points (np.ndarray): The predicted head positions from P2PNeXt. 
        Shape [Num_Pred_Points, 2]
        confidence_scores (np.ndarray): The confidence scores to the predicted head positions.
        Shape [Num_Pred_Points, ]

    :returns: A tuple of numpy arrays. First the coordinates of the sorted predicted points. 
    Second the sorted confidence scores of the predicted points.
    '''
    
    if predicted_points.size == 0:
        return np.array([]), np.array([])
        
    idx = np.argsort(confidence_scores)[::-1]
    
    sorted_prediction_points = predicted_points[idx]
    sorted_confidence_scores = confidence_scores[idx]

    return sorted_prediction_points, sorted_confidence_scores


def _match_predictions_to_ground_truths_for_nap(prediction_points: np.ndarray, gt_points: np.ndarray, 
                                                k: int=3, threshold: float=0.5, log: bool=False):
    
    matched = np.zeros(len(gt_points), dtype=bool)
    tp = np.zeros(len(prediction_points))
    fp = np.zeros(len(prediction_points))

    if gt_points.shape == 0:
        if not prediction_points.shape == 0:
            fp[:] = 1
            return tp, fp
    
    if prediction_points.shape == 0:
        return tp, fp
    
    k = min(k, len(gt_points) - 1)
    kd_tree = KDTree(gt_points)

    if k > 0:
        k_nn_distances, _ = kd_tree.query(gt_points, k=k+1)
        d_knn = np.mean(k_nn_distances[:, 1:], axis=1)
    else:
        d_knn = np.full(len(gt_points), 100.0)

    for i, pred_point in enumerate(prediction_points):
        min_dist = float('inf')
        best_gt_index = -1
        gt_idx = kd_tree.query_ball_point(pred_point, r=np.max(d_knn) * threshold)

        for j in gt_idx:
            if matched[j]:
                continue

            dist = np.linalg.norm(pred_point - gt_points[j])
            if dist < d_knn[j] * threshold and dist < min_dist:
                min_dist = dist
                best_gt_index = j

        if best_gt_index != -1:
            matched[best_gt_index] = True
            tp[i] = 1
            if log:
                logger.info(f"Predicted Point {pred_point} matched with Ground Truth {gt_points[best_gt_index]} with minimum distance {min_dist} and {d_knn[best_gt_index] * threshold}")
        else:
            fp[i] = 1
            if log:
                logger.info(f"Predicted Point {pred_point} did not match with any Ground Truth {min_dist}")
    
    return tp, fp


def _match_predictions_to_ground_truths_for_ap(prediction_points: np.ndarray, gt_points: np.ndarray, 
                                               threshold=4, log: bool=False):
    
    matched = np.zeros(len(gt_points), dtype=bool)
    tp = np.zeros(len(prediction_points))
    fp = np.zeros(len(prediction_points))

    if gt_points.shape == 0:
        if not prediction_points.shape == 0:
            fp[:] = 1
            return tp, fp
    
    if prediction_points.shape == 0:
        return tp, fp
    
    kd_tree = KDTree(gt_points)

    for i, pred_point in enumerate(prediction_points):
        min_dist = float('inf')
        best_gt_index = -1
        gt_idx = kd_tree.query_ball_point(pred_point, r=threshold)

        for j in gt_idx:
            if matched[j]:
                continue

            dist = np.linalg.norm(pred_point - gt_points[j])
            if dist < threshold and dist < min_dist:
                min_dist = dist
                best_gt_index = j

        if best_gt_index != -1:
            matched[best_gt_index] = True
            tp[i] = 1
            if log:
                logger.info(f"Predicted Point {pred_point} matched with Ground Truth {gt_points[best_gt_index]} with minimum distance {min_dist} and {threshold}")
        else:
            fp[i] = 1
            if log:
                logger.info(f"Predicted Point {pred_point} did not match with any Ground Truth {min_dist}")
    
    return tp, fp


def _calculate_auc_score(tp: np.ndarray, fp: np.ndarray, total_amount_gt_points: int):
    if total_amount_gt_points == 0:
        return 1.0 if len(tp) == 0 else 0.0
    
    if len(tp) == 0:
        return 0.0

    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_amount_gt_points

    precisions = np.concatenate([[1], precisions])
    recalls = np.concatenate([[0], recalls])

    return auc(recalls, precisions)


def get_nap_score(predicted_points: np.ndarray, predicted_confidences: np.ndarray, 
                  target_points: np.ndarray, k: int=3, threshold:int=0.5, logging:bool=False):
    
    amount_of_gt_points = target_points.shape[0]

    sorted_pred_points, _ = _sort_outputs_by_confidence_scores(predicted_points, predicted_confidences)
    tp, fp = _match_predictions_to_ground_truths_for_nap(sorted_pred_points, target_points, k, 
                                                         threshold, logging)
    nap_score = _calculate_auc_score(tp, fp, amount_of_gt_points)
    return nap_score, threshold, k


def get_ap_score(predicted_points: np.ndarray, predicted_confidences: np.ndarray, 
                 target_points: np.ndarray, threshold: int=4, logging: bool=False):
    
    amount_of_gt_points = target_points.shape[0]

    sorted_pred_points, _ = _sort_outputs_by_confidence_scores(predicted_points, predicted_confidences)
    tp, fp = _match_predictions_to_ground_truths_for_ap(sorted_pred_points, target_points, threshold, logging)
    ap_score = _calculate_auc_score(tp, fp, amount_of_gt_points)
    return ap_score, threshold