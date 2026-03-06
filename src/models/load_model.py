from typing import Any

import timm
import torch
from torch import nn

DINO_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/"


class ModelLoader:
    def __init__(
            self,
            cfg: dict[Any, str]
    ):
        self.cfg = cfg
        self.model_checkpoint = self.cfg["model"]["checkpoint"]

    def load(self) -> tuple[nn.Module, int]:

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
