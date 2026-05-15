from typing import Any

import torch
import torch.nn as nn
from torch import optim

from src.train_eval.solver_classification import SolverClassification
from src.utils.dino_utils import cosine_scheduler, cancel_gradients_layer_x, clip_gradients
from src.utils.optim_utils import get_maybe_fused_params_for_submodel


class SupervisedDINO(SolverClassification):
    def __init__(
            self,
            model: nn.Module,
            cfg: dict[str, Any],
            iter_per_epoch: int
    ):
        super().__init__(model=model, cfg=cfg)

        self.automatic_optimization = False

        # params
        self.iter_per_epoch = iter_per_epoch

        # loss
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
        }, on_step=False, on_epoch=True, prog_bar=True)
        preds = torch.argmax(predictions, dim=1)
        self.train_preds.append(preds.detach().cpu())
        self.train_targets.append(targets.detach().cpu())
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

    def test_step(self, batch, batch_idx):
        predictions, targets = self._common_step(batch, batch_idx)
        preds = torch.argmax(predictions, dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        return
