import torch
from sklearn.metrics import confusion_matrix

from src.utils.metrics import accuracy, sensitivity, specificity
from src.utils.metrics_model import CMMetrics


class MetricCalculator:

    def __init__(self):
        self.targets = []
        self.predictions = []
        self.individual_cms = None
        self.total_num = 0

    def _is_individual_cms_not_none(self):
        if self.targets is None or self.predictions is None:
            raise ValueError("Targets and predictions must be provided before computing metrics.")
        if self.individual_cms is None:
            self.individual_cms = self.compute()

    def _get_individual_cms(
            self,
            target: torch.Tensor,
            predictions: torch.Tensor
    ) -> dict[int, CMMetrics]:
        labels = [0, 1, 2, 3]
        cm = confusion_matrix(target, predictions, labels=labels)
        metrics = {}
        for label in labels:
            metrics[label] = self._get_cm_per_class(cm, label)
        return metrics

    def _get_cm_per_class(
            self,
            cm,
            label: int
    ) -> CMMetrics:
        tp = cm[label, label]
        fn = cm[label, :].sum() - tp
        fp = cm[:, label].sum() - tp
        tn = cm.sum() - tp - fn - fp
        total = tp + fn
        return CMMetrics(total=total, tp=tp, fp=fp, fn=fn, tn=tn)

    def update(
            self,
            target: torch.Tensor,
            predictions: torch.Tensor
    ):
        self.targets.append(target)
        self.predictions.append(predictions)
        self.total_num += target.shape[0]
        self._is_individual_cms_not_none()

    def reset(self):
        self.targets = []
        self.predictions = []
        self.individual_cms = None
        self.total_num = 0

    def compute(
            self
    ) -> dict[int, CMMetrics]:
        all_targets = torch.cat(self.targets)
        all_predictions = torch.cat(self.predictions)
        return self._get_individual_cms(all_targets, all_predictions)

    def get_micro_avg_accuracy(
            self
    ) -> float:
        self._is_individual_cms_not_none()
        total_tp = sum(cm.tp for cm in self.individual_cms.values())
        total_tn = sum(cm.tn for cm in self.individual_cms.values())
        total = sum(cm.total for cm in self.individual_cms.values())
        if total == 0:
            return 0.0
        return (total_tp + total_tn) / total

    def get_macro_avg_accuracy(
            self
    ) -> float:
        self._is_individual_cms_not_none()
        accuracies = [accuracy(cm) for cm in self.individual_cms.values()]
        if len(accuracies) == 0:
            return 0.0
        return sum(accuracies) / len(accuracies)

    def get_weighted_avg_accuracy(
            self
    ) -> float:
        self._is_individual_cms_not_none()
        total = sum(cm.total for cm in self.individual_cms.values())
        if total == 0:
            return 0.0
        weighted_accuracy = 0.0
        for cm in self.individual_cms.values():
            acc = accuracy(cm)
            weight = (cm.tp + cm.fn) / total
            weighted_accuracy += acc * weight
        return weighted_accuracy

    def get_micro_avg_sensitivity(
            self
    ) -> float:
        self._is_individual_cms_not_none()
        total_tp = sum(cm.tp for cm in self.individual_cms.values())
        total_fn = sum(cm.fn for cm in self.individual_cms.values())
        if total_tp + total_fn == 0:
            return 0.0
        return total_tp / (total_tp + total_fn)

    def get_macro_avg_sensitivity(
            self
    ) -> float:
        self._is_individual_cms_not_none()
        sensitivities = [sensitivity(cm) for cm in self.individual_cms.values()]
        if len(sensitivities) == 0:
            return 0.0
        return sum(sensitivities) / len(sensitivities)

    def get_weighted_avg_sensitivity(
            self
    ) -> float:
        self._is_individual_cms_not_none()
        total = sum(cm.total for cm in self.individual_cms.values())
        if total == 0:
            return 0.0
        weighted_sensitivity = 0.0
        for cm in self.individual_cms.values():
            sens = sensitivity(cm)
            weight = cm.total / total
            weighted_sensitivity += sens * weight
        return weighted_sensitivity

    def get_micro_avg_specificity(
            self
    ) -> float:
        self._is_individual_cms_not_none()
        total_tn = sum(cm.tn for cm in self.individual_cms.values())
        total_fp = sum(cm.fp for cm in self.individual_cms.values())
        if total_tn + total_fp == 0:
            return 0.0
        return total_tn / (total_tn + total_fp)

    def get_macro_avg_specificity(
            self
    ) -> float:
        self._is_individual_cms_not_none()
        specificities = [specificity(cm) for cm in self.individual_cms.values()]
        if len(specificities) == 0:
            return 0.0
        return sum(specificities) / len(specificities)

    def get_weighted_avg_specificity(
            self
    ) -> float:
        self._is_individual_cms_not_none()
        total = sum(cm.total for cm in self.individual_cms.values())
        if total == 0:
            return 0.0
        weighted_specificity = 0.0
        for cm in self.individual_cms.values():
            spec = specificity(cm)
            weight = cm.total / total
            weighted_specificity += spec * weight
        return weighted_specificity

