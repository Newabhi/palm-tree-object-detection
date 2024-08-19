import numpy as np


def count_palm_trees(predictions):

    return sum((pred['labels'] == 1).sum().item() for pred in predictions)

def compute_metrics(predictions, targets):

    pred_counts = np.array([count_palm_trees([pred]) for pred in predictions])
    gt_counts = np.array([targ.item() for targ in targets])
    
    mae = np.mean(np.abs(pred_counts - gt_counts))
    rmse = np.sqrt(np.mean((pred_counts - gt_counts) ** 2))
    
    return mae, rmse
