from typing import Any

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.train_eval.solver_classification import SolverClassification


class SolverLinearProbing(SolverClassification):
    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            cfg: dict[str, Any]
    ):
        super().__init__(model=model, num_classes=num_classes, cfg=cfg)

        self.loss_function = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer_name = self.cfg['optimizer']['name']
        self.lr = self.cfg['optimizer']['lr']
        self.momentum = self.cfg['optimizer']['momentum']

    def training_step(self, batch, batch_idx):
        predictions, targets = self._common_step(batch, batch_idx)
        loss = self.loss_function(predictions, targets)
        self.log_dict({
            "train/train_loss": loss,
        }, on_step=False, on_epoch=True, prog_bar=True)
        preds = torch.argmax(predictions, dim=1)
        self.train_preds.append(preds.detach().cpu())
        self.train_targets.append(targets.detach().cpu())
        return {"loss": loss, "predictions": predictions, "targets": targets}

    def validation_step(self, batch, batch_idx):
        predictions, targets = self._common_step(batch, batch_idx)
        loss = self.loss_function(predictions, targets)
        self.log_dict({
            "val/val_loss": loss,
        })
        preds = torch.argmax(predictions, dim=1)
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())
        return

    def test_step(self, batch, batch_idx):
        predictions, targets = self._common_step(batch, batch_idx)
        preds = torch.argmax(predictions, dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        return

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg["epochs"])
        return [optimizer], [{"scheduler": decay_scheduler, "interval": "epoch"}]
