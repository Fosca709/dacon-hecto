import os
from pathlib import Path
from typing import Self

import open_clip
import safetensors
import torch
import torch.nn as nn

from .config import BaseConfig


class ClassifierConfig(BaseConfig):
    hidden_dim: int
    num_classes: int
    image_size: tuple[int, int]
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    model_name: str | None = None


class Classifier(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        config: ClassifierConfig,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.config = config
        self.vision_encoder = vision_encoder
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_classes),
        ).to(device)

        self.normalize_mean = config.normalize_mean
        self.normalize_std = config.normalize_std

    def forward(self, x):
        x = self.vision_encoder(x)
        x = self.fc(x)
        return x

    @classmethod
    def from_open_clip(
        cls,
        model_name: str,
        pretrained: str | None,
        cache_dir: Path,
        device: str | torch.device,
        image_size: tuple[int] | None,
        num_classes: int,
        model_type: str | None = None,
        hidden_dim: int | None = None,
    ) -> Self:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
            device=device,
            force_image_size=image_size,
        )

        vision_encoder = clip_model.visual

        if hidden_dim is None:
            if model_type == "vit":
                hidden_dim = vision_encoder.output_dim
            else:
                raise Exception("`hidden_dim` must be set when using this `model_type`")

        normalize_mean, normalize_std = None, None
        for t in preprocess.transforms:
            t_name = t.__class__.__name__
            if t_name == "Normalize":
                normalize_mean = t.mean
                normalize_std = t.std

            elif t_name == "CenterCrop" and image_size is None:
                image_size = t.size

        if None in [normalize_mean, normalize_std, image_size]:
            raise Exception("Incomplete config: missing mean, std, or image size.")

        config = ClassifierConfig(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            image_size=image_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            model_name=model_name,
        )

        return Classifier(
            vision_encoder=vision_encoder,
            config=config,
            device=device,
        )

    def save_model(self, model_path: Path):
        os.makedirs(model_path, exist_ok=True)

        # save config
        self.config.__save__(model_path / "config.yaml")

        # save model
        safetensors.torch.save_model(self, model_path / "model.safetensors")

    @classmethod
    def from_pretrained(cls, model_path: Path, device: str | torch.device) -> Self:
        # load config
        config = ClassifierConfig.__load__(model_path / "config.yaml")

        # initialize vision encoder
        clip_model = open_clip.create_model(
            model_name=config.model_name, pretrained=None, device=device, force_image_size=config.image_size
        )
        vision_encoder = clip_model.visual

        # initialize model
        classifier = Classifier(vision_encoder=vision_encoder, config=config, device=device)

        # load weights
        safetensors.torch.load_model(classifier, model_path / "model.safetensors")
        return classifier

    def set_encoder_grad(self, mode: bool):
        for param in self.vision_encoder.parameters():
            param.requires_grad_(mode)
