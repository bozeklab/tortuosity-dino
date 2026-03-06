from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import wandb
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.metric_calculator import MetricCalculator
from src.utils.metrics import accuracy, sensitivity, specificity


class SolverLinearProbing(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            cfg: dict[str, Any]
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.model = model

        # params
        self.cfg = cfg
        self.loss_function = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer_name = self.cfg['optimizer']['name']
        self.lr = self.cfg['optimizer']['lr']
        self.momentum = self.cfg['optimizer']['momentum']

        # metrics
        self.best_val_accuracy = 0.0
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # val
        self.val_preds = []
        self.val_targets = []

        # test
        self.test_preds = []
        self.test_targets = []
        self.metric_calculator = MetricCalculator()

    def forward(self, x):
        out = self.model(x)
        return out

    def _common_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self.model(data)
        loss = self.loss_function(predictions, targets)
        return loss, predictions, targets

    def training_step(self, batch, batch_idx):
        loss, predictions, targets = self._common_step(batch, batch_idx)
        self.log_dict({
            "train/train_loss": loss,
            "train/train_accuracy": self.accuracy(predictions, targets),
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "predictions": predictions, "targets": targets}

    def validation_step(self, batch, batch_idx):
        loss, predictions, targets = self._common_step(batch, batch_idx)
        self.log_dict({
            "val/val_loss": loss,
        })
        preds = torch.argmax(predictions, dim=1)
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())
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

    def test_step(self, batch, batch_idx):
        loss, predictions, targets = self._common_step(batch, batch_idx)
        preds = torch.argmax(predictions, dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
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

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg["epochs"])
        return [optimizer], [{"scheduler": decay_scheduler, "interval": "epoch"}]
