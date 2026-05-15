from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import v2 as T

import src.models.dino.vision_transformers  as vits
from src.models.load_model import ModelLoader
from src.train_eval.supervised_dino_ft import SupervisedDINO


def apply_mask(image: NDArray, mask: NDArray, color: tuple[float, float, float], alpha: float=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def normalize_img(img, uint: bool=True):
    min_val = img.min()
    max_val = img.max()
    img_norm = (img - min_val) / (max_val - min_val)
    if uint:
        img_uint8 = (img_norm * 255).byte()
        return img_uint8
    return img_norm


def overlay_attention(
        img: NDArray,
        attention_map: NDArray,
        color: tuple[float, float, float]=(1.0, 0.0, 0.0),
        alpha: float=0.6
):
    attention_map = normalize_img(attention_map, uint=False)
    masked_image = img.astype(np.uint32).copy()
    masked_image = apply_mask(masked_image, attention_map, color, alpha)
    return masked_image


def prepare_img(img: Image.Image, patch_size: int):
    transform = T.Compose([
        T.Resize(304),
        T.Grayscale(num_output_channels=3),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.339], [0.138]),
    ])
    img = transform(img)
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    return img


def get_attentions(
        model: nn.Module,
        img_sample: torch.Tensor,
        threshold: float=0.6,
        patch_size: int=16,
):
    """
    :param model:
    :return:
        - Number of Attention Heads
        - Attention Values
        - Thresholded Binary Attention Values
    """
    w_featmap = img_sample.shape[-2] // patch_size
    h_featmap = img_sample.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img_sample)

    num_heads = attentions.shape[1] # number of heads

    # we keep only the output patch attention -> CLS Token
    attentions = attentions[0, :, 0, 1:].reshape(num_heads, -1) # N, num_heads, CLS + num_patches Query, CLS + num_patches Keys

    # we keep only a certain percentage of the mass
    att_vals, idx = torch.sort(attentions)
    att_vals /= torch.sum(att_vals, dim=1, keepdim=True)
    cumval = torch.cumsum(att_vals, dim=1) # through sorting this becomes the accumulation of the mass
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(num_heads):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(num_heads, w_featmap, h_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(num_heads, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    return num_heads, attentions, th_attn

def get_model(
        cfg: dict,
        fine_tuned: bool=False,
        patch_size: int=16,
        model_checkpoint: Path=None,
):
    model = vits.__dict__['vit_base'](patch_size=patch_size, num_classes=0)
    if fine_tuned:
        LIGHTNING_MODULE = SupervisedDINO
        model_loader = ModelLoader(cfg, model_checkpoint)

        classifier, _ = model_loader.load(LIGHTNING_MODULE)
        model.load_state_dict(classifier.model.feature_extractor.state_dict())
    else:
        model_loader = ModelLoader(cfg)
        state_dict = model_loader.load_state_dict()
        msg = model.load_state_dict(state_dict, strict=False)
        print("=> loaded backbone from checkpoint '{}' with msg {}".format(cfg["model"]["checkpoint"], msg))

    model.head = torch.nn.Identity()
    return model

def matplotlib_colors(N: int, cmap_name: str="tab10"):
    """
    Generate N distinct matplotlib colors.
    Returns list of RGB tuples.
    """
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N)[:3] for i in range(N)]
    return colors
