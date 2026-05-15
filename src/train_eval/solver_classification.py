from typing import Any

import lightning as L
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import confusion_matrix

from src.utils.metric_calculator import MetricCalculator
from src.utils.metrics import accuracy, sensitivity, specificity


class SolverClassification(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            cfg: dict[str, Any]
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.model = model

        # params
        self.cfg = cfg

        # metrics
        self.best_val_accuracy = 0.0
        self.metric_calculator = MetricCalculator()

        ## train
        self.train_preds = []
        self.train_targets = []

        ## val
        self.val_preds = []
        self.val_targets = []

        ## test
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        out = self.model(x)
        return out

    def _common_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self.model(data)
        return predictions, targets

    def on_train_epoch_end(self):
        preds = torch.cat(self.train_preds, dim=0)
        targets = torch.cat(self.train_targets, dim=0)
        self.metric_calculator.reset()
        self.metric_calculator.update(targets, preds)
        train_acc = self.metric_calculator.get_weighted_avg_accuracy()
        self.log_dict({
            "train/train_weighted_accuracy": train_acc,
            "train/train_weighted_specificity": self.metric_calculator.get_weighted_avg_specificity(),
            "train/train_weighted_sensitivity": self.metric_calculator.get_weighted_avg_sensitivity(),
        }, on_step=False, on_epoch=True, prog_bar=True)
        self.train_preds.clear()
        self.train_targets.clear()
        return

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)
        self.metric_calculator.reset()
        self.metric_calculator.update(targets, preds)
        val_acc = self.metric_calculator.get_weighted_avg_accuracy()
        self.log_dict({
            "val_weighted_accuracy": val_acc,
            "val/val_weighted_accuracy": val_acc,
            "val/val_weighted_specificity": self.metric_calculator.get_weighted_avg_specificity(),
            "val/val_weighted_sensitivity": self.metric_calculator.get_weighted_avg_sensitivity(),
            "val/val_accuracy_grade1": accuracy(self.metric_calculator.individual_cms[0]),
            "val/val_accuracy_grade2": accuracy(self.metric_calculator.individual_cms[1]),
            "val/val_accuracy_grade3": accuracy(self.metric_calculator.individual_cms[2]),
            "val/val_accuracy_grade4": accuracy(self.metric_calculator.individual_cms[3]),
            "val/val_specificity_grade1": specificity(self.metric_calculator.individual_cms[0]),
            "val/val_specificity_grade2": specificity(self.metric_calculator.individual_cms[1]),
            "val/val_specificity_grade3": specificity(self.metric_calculator.individual_cms[2]),
            "val/val_specificity_grade4": specificity(self.metric_calculator.individual_cms[3]),
            "val/val_sensitivity_grade1": sensitivity(self.metric_calculator.individual_cms[0]),
            "val/val_sensitivity_grade2": sensitivity(self.metric_calculator.individual_cms[1]),
            "val/val_sensitivity_grade3": sensitivity(self.metric_calculator.individual_cms[2]),
            "val/val_sensitivity_grade4": sensitivity(self.metric_calculator.individual_cms[3]),
        }, on_step=False, on_epoch=True, prog_bar=True)
        self.val_preds.clear()
        self.val_targets.clear()

        self.best_val_accuracy = max(val_acc, self.best_val_accuracy)
        self.log_dict({
            "val/best_val_accuracy": self.best_val_accuracy,
        }, on_step=False, on_epoch=True, prog_bar=True)
        return

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds, dim=0)
        targets = torch.cat(self.test_targets, dim=0)
        self.metric_calculator.reset()
        self.metric_calculator.update(targets, preds)
        self.log_dict({
            "test/test_weighted_accuracy": self.metric_calculator.get_weighted_avg_accuracy(),
            "test/test_weighted_specificity": self.metric_calculator.get_weighted_avg_specificity(),
            "test/test_weighted_sensitivity": self.metric_calculator.get_weighted_avg_sensitivity(),
            "test/test_accuracy_grade1": accuracy(self.metric_calculator.individual_cms[0]),
            "test/test_accuracy_grade2": accuracy(self.metric_calculator.individual_cms[1]),
            "test/test_accuracy_grade3": accuracy(self.metric_calculator.individual_cms[2]),
            "test/test_accuracy_grade4": accuracy(self.metric_calculator.individual_cms[3]),
            "test/test_specificity_grade1": specificity(self.metric_calculator.individual_cms[0]),
            "test/test_specificity_grade2": specificity(self.metric_calculator.individual_cms[1]),
            "test/test_specificity_grade3": specificity(self.metric_calculator.individual_cms[2]),
            "test/test_specificity_grade4": specificity(self.metric_calculator.individual_cms[3]),
            "test/test_sensitivity_grade1": sensitivity(self.metric_calculator.individual_cms[0]),
            "test/test_sensitivity_grade2": sensitivity(self.metric_calculator.individual_cms[1]),
            "test/test_sensitivity_grade3": sensitivity(self.metric_calculator.individual_cms[2]),
            "test/test_sensitivity_grade4": sensitivity(self.metric_calculator.individual_cms[3]),
        }, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({
            "test/test_confusion_matrix_table":
                wandb.Table(
                    data=confusion_matrix(targets.detach().numpy(), preds.detach().numpy()).tolist(),
                    columns=['grade 1', 'grade 2', 'grade 3', 'grade 4']
                ),
        })
        wandb.log({
            "test/test_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=targets.detach().numpy(),
                preds=preds.detach().numpy(),
                class_names=['grade 1', 'grade 2', 'grade 3', 'grade 4'])
        })
        self.test_preds.clear()
        self.test_targets.clear()
        return
