import os
from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from src.dataloaders.cnf_datamodule import CNFDataModule
from src.dataloaders.cnf_dataset import CNFDataset
from src.dataloaders.transforms import TransformsSupervisedLearningTrain, TestTransforms
from src.models.linear_classifier import LinearClassifier
from src.models.load_model import ModelLoader
from src.train_eval.supervised_dino_ft import SupervisedDINO
from src.utils.read_write import read_yaml

CWD = Path(__file__).parent
DATA_DIR = CWD / "data"
CONFIG_PATH = CWD / "configs" / "supervised_dino.yaml"
PROJECT_NAME = "corn1500_dino_sl_clean"
NUM_CLASSES = 4
VAL_SIZE = 0.3

MODEL_CHECKPOINT_DIR = DATA_DIR.parent / "model_checkpoints" / f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{PROJECT_NAME}"

RANDOM_SEED = 42
L.seed_everything(RANDOM_SEED)


def main():
    wandb_settings = wandb.Settings(
        console="wrap",
        show_errors=True,
        show_warnings=True,
        show_info=True,
    )

    cfg = read_yaml(CONFIG_PATH)
    wandb_logger = WandbLogger(project=PROJECT_NAME, log_model=False, settings=wandb_settings)

    train_transforms = TransformsSupervisedLearningTrain(output_dim=cfg["model"]["output_size"])
    test_transforms = TestTransforms(output_dim=cfg["model"]["output_size"])

    ## load data
    data_module = CNFDataModule(
        data_dir=DATA_DIR,
        batch_size=cfg['batch_size'],
        num_workers=3,
        val_size=VAL_SIZE,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        dataset=CNFDataset,
    )

    ## load model
    model_loader = ModelLoader(cfg)
    backbone, embedding_size = model_loader.load()

    model = LinearClassifier(
        feature_extractor=backbone,
        embedding_dim=embedding_size,
        num_classes=NUM_CLASSES,
    )

    iter_per_epoch = int(np.ceil(1250 * (1-VAL_SIZE) / cfg["batch_size"]))
    solver = SupervisedDINO(model=model, cfg=cfg, iter_per_epoch=iter_per_epoch)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # model checkpoint callback
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_weighted_accuracy',
        mode='max',
        save_top_k=1,
        save_last=True,
        dirpath=MODEL_CHECKPOINT_DIR,
        filename='{epoch}-{val_weighted_accuracy:.2f}'
    )

    trainer = L.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=cfg["epochs"],
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    print(f"Optimization Turned On: {solver.automatic_optimization}")
    trainer.fit(model=solver, datamodule=data_module)
    trainer.validate(model=solver, datamodule=data_module)
    trainer.test(model=solver, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
