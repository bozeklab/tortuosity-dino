from typing import Any

import timm
import torch
from torch import nn
import lightning as L

DINO_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/"


class ModelLoader:
    def __init__(
            self,
            cfg: dict[Any, str],
            checkpoint: str = None
    ):
        self.cfg = cfg
        self.model_checkpoint = checkpoint if checkpoint else self.cfg["model"]["checkpoint"]

    def load(self, lightning_module: L.LightningModule = None) -> tuple[nn.Module, int]:

        if lightning_module is not None:
            print("Use Pretrained Model with Lightning Module")
            model_loaded = lightning_module.load_from_checkpoint(self.model_checkpoint)
            return model_loaded, None

        else:
            feature_extractor_model = timm.create_model(
                model_name=self.cfg["model"]["backbone"],
                patch_size=self.cfg["model"]["patch_size"],
                dynamic_img_size=True,
                pretrained=False,
                drop_path_rate=self.cfg["model"]["drop_path_rate"],
            )
            state_dict = torch.hub.load_state_dict_from_url(DINO_CHECKPOINT_URL + self.model_checkpoint, map_location="cpu", weights_only=False)
            model_state_dict = state_dict["teacher"]
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
            model_state_dict = {k.replace("backbone.", ""): v for k, v in model_state_dict.items()}
            msg = feature_extractor_model.load_state_dict(model_state_dict, strict=False)
            print("=> loaded backbone from checkpoint '{}' with msg {}".format(self.model_checkpoint, msg))
            feature_extractor_model.head = torch.nn.Identity()
            embedding_size = feature_extractor_model.embed_dim

            return feature_extractor_model, embedding_size

    def load_state_dict(self):
        state_dict = torch.hub.load_state_dict_from_url(
            DINO_CHECKPOINT_URL + self.model_checkpoint,
            map_location="cpu",
            weights_only=False
        )
        model_state_dict = state_dict["teacher"]
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model_state_dict = {k.replace("backbone.", ""): v for k, v in model_state_dict.items()}
        return model_state_dict
