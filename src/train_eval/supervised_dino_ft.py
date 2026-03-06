from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import wandb
from sklearn.metrics import confusion_matrix
from torch import optim

from src.utils.dino_utils import cosine_scheduler, cancel_gradients_layer_x, clip_gradients
from src.utils.metric_calculator import MetricCalculator
from src.utils.metrics import accuracy, sensitivity, specificity
from src.utils.optim_utils import get_maybe_fused_params_for_submodel


class SupervisedDINO(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            cfg: dict[str, Any],
            iter_per_epoch: int
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.model = model
        self.automatic_optimization = False

        # params
        self.cfg = cfg
        self.iter_per_epoch = iter_per_epoch
        self.train_loss_function = nn.CrossEntropyLoss(label_smoothing=cfg['loss']['label_smoothing'])
        self.val_loss_function = nn.CrossEntropyLoss()

        # optimizer
        self.lr = self.cfg['optimizer']['lr']
        self.min_lr = self.cfg['optimizer']['min_lr']
        self.weight_decay_start = self.cfg['optimizer']['weight_decay']['start']
        self.weight_decay_end = self.cfg['optimizer']['weight_decay']['end']
        self.warmup_epochs = self.cfg['optimizer']['warmup_epochs']
        self.frozen_layers = self.cfg['model']['frozen_layers']
        self.frozen_epochs = self.cfg['model']['frozen_epochs']
        self.lr_schedule = None
        self.weight_decay_schedule = None
        self.configure_schedulers()

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

    def configure_optimizers(self):
        optimizer = optim.AdamW(get_maybe_fused_params_for_submodel(self.cfg, self.model))
        return optimizer

    def configure_schedulers(self):
        self.lr_schedule = cosine_scheduler(
            base_value=self.lr * self.cfg['batch_size'] / 256.,
            final_value=self.min_lr,
            epochs=self.cfg['epochs'],
            niter_per_ep=self.iter_per_epoch,
            warmup_epochs=self.warmup_epochs,
        )
        self.weight_decay_schedule = cosine_scheduler(
            base_value=self.weight_decay_start,
            final_value=self.weight_decay_end,
            epochs=self.cfg['epochs'],
            niter_per_ep=self.iter_per_epoch,
        )

    def apply_optim_scheduler(self, lr, wd):
        opt = self.optimizers()
        for param_group in opt.param_groups:
            lr_multiplier = param_group["lr_multiplier"]
            wd_multiplier = param_group["wd_multiplier"]
            param_group["weight_decay"] = wd * wd_multiplier
            param_group["lr"] = lr * lr_multiplier

    def forward(self, x):
        out = self.model(x)
        return out

    def _common_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self.model(data)
        return predictions, targets

    def training_step(self, batch, batch_idx):
        lr = self.lr_schedule[self.global_step]
        weight_decay = self.weight_decay_schedule[self.global_step]
        self.apply_optim_scheduler(lr=lr, wd=weight_decay)

        opt = self.optimizers()
        opt.zero_grad()

        predictions, targets = self._common_step(batch, batch_idx)
        loss = self.train_loss_function(predictions, targets)

        self.manual_backward(loss)
        clip_gradients(self.model)

        for i in range(self.frozen_layers):
            block_name = f"blocks.{i}."
            cancel_gradients_layer_x(
                epoch=self.current_epoch,
                model=self.model,
                freeze_duration=self.frozen_epochs,
                layer_name=block_name,
            )

        opt.step()

        self.log_dict({
            "train/train_loss": loss,
            "train/train_accuracy": self.accuracy(predictions, targets),
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "predictions": predictions, "targets": targets}

    def validation_step(self, batch, batch_idx):
        predictions, targets = self._common_step(batch, batch_idx)
        loss = self.val_loss_function(predictions, targets)
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
            "best_val_accuracy": self.best_val_accuracy,
        }, on_step=False, on_epoch=True, prog_bar=True)
        return

    def test_step(self, batch, batch_idx):
        predictions, targets = self._common_step(batch, batch_idx)
        preds = torch.argmax(predictions, dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        return

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds, dim=0)
        targets = torch.cat(self.test_targets, dim=0)
        self.metric_calculator.reset()
        self.metric_calculator.update(targets, preds)
        preds = preds.to(self.device)
        targets = targets.to(self.device)
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
                )
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
