from pathlib import Path
from datetime import datetime
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

from src.dataloaders.cnf_datamodule import CNFDataModule
from src.dataloaders.cnf_dataset import CNFDataset
from src.dataloaders.transforms import TransformsLinearProbingTrain, TestTransforms
from src.models.linear_classifier import LinearClassifier
from src.models.load_model import ModelLoader
from src.train_eval.solver_linear_probing import SolverLinearProbing
from src.utils.read_write import read_yaml

CWD = Path(__file__).parent
DATA_DIR = CWD / 'data'
CONFIG_PATH = CWD / "configs" / "ssl_linear_probing.yaml"
PROJECT_NAME = "CORN1500_linear_probing_clean"
NUM_CLASSES = 4

wandb_settings = wandb.Settings(
    console="wrap",
    show_errors=True,
    show_warnings=True,
    show_info=True,
)

WANDB_LOGGER = WandbLogger(project=PROJECT_NAME, log_model=False, settings=wandb_settings)
NUM_MODEL_CHECKPOINTS = 1
MODEL_CHECKPOINT_DIR = DATA_DIR.parent / "model_checkpoints" / f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{PROJECT_NAME}"

RANDOM_SEED = 42
L.seed_everything(RANDOM_SEED)


def main():
    cfg = read_yaml(CONFIG_PATH)

    train_transforms = TransformsLinearProbingTrain(output_dim=cfg["model"]["output_size"])
    test_transforms = TestTransforms(output_dim=cfg["model"]["output_size"])

    data_module = CNFDataModule(
        data_dir=DATA_DIR,
        batch_size=cfg['batch_size'],
        num_workers=6,
        val_size=0.3,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        dataset=CNFDataset,
    )

    ## load model
    model_loader = ModelLoader(cfg=cfg)
    feature_extractor_model, embedding_size = model_loader.load()

    feature_extractor_model.eval()

    classifier_model = LinearClassifier(
        feature_extractor=feature_extractor_model,
        embedding_dim=embedding_size,
        num_classes=NUM_CLASSES,
    )
    for param in classifier_model.feature_extractor.parameters():
        param.requires_grad = False

    solver = SolverLinearProbing(model=classifier_model, cfg=cfg)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    print(f"Saving model to {MODEL_CHECKPOINT_DIR}")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_weighted_accuracy',
        mode='max',
        save_top_k=NUM_MODEL_CHECKPOINTS,
        save_last=True,
        dirpath=MODEL_CHECKPOINT_DIR,
        filename='{epoch}-{val_weighted_accuracy:.2f}'
    )

    trainer = L.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=cfg['epochs'],
        logger=WANDB_LOGGER,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    trainer.fit(model=solver, datamodule=data_module)
    trainer.validate(model=solver, datamodule=data_module)
    trainer.test(model=solver, dataloaders=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
