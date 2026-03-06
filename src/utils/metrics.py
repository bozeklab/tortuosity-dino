from src.utils.metrics_model import CMMetrics


def sensitivity(
        cm_metrics: CMMetrics
) -> float:
    """Sensitivity = TP / (TP + FN)"""
    if cm_metrics.tp == 0:
        return 0.0
    return cm_metrics.tp / (cm_metrics.tp + cm_metrics.fn)

def specificity(
        cm_metrics: CMMetrics
) -> float:
    """Specificity = TN / (TN + FP)"""
    if cm_metrics.tn == 0:
        return 0.0
    return cm_metrics.tn / (cm_metrics.tn + cm_metrics.fp
)

def accuracy(
        cm_metrics: CMMetrics
) -> float:
    """Accuracy = (TP + TN) / (TP + TN + FP + FN)"""
    total = cm_metrics.tp + cm_metrics.tn + cm_metrics.fp + cm_metrics.fn
    if total == 0:
        return 0.0
    return (cm_metrics.tp + cm_metrics.tn) / total

def precision(
        cm_metrics: CMMetrics
) -> float:
    """Precision = TP / (TP + FP)"""
    if cm_metrics.tp == 0:
        return 0.0
    return cm_metrics.tp / (cm_metrics.tp + cm_metrics.fp)

def recall(
        cm_metrics: CMMetrics
) -> float:
    """Recall = TP / (TP + FN)"""
    if cm_metrics.tp == 0:
        return 0.0
    return cm_metrics.tp / (cm_metrics.tp + cm_metrics.fn
)

def f1(
        cm_metrics: CMMetrics
) -> float:
    """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)"""
    prec = precision(cm_metrics)
    rec = recall(cm_metrics)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec
)
