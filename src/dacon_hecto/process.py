from typing import Any, Callable

import albumentations as A
import cv2
import numpy as np
import torch

type ImageProcessor = Callable[[Any], torch.Tensor]


def load_image(image_path) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_default_processor(
    image_size: tuple[int, int],
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
    interpolation=cv2.INTER_AREA,
) -> ImageProcessor:
    h, w = image_size
    assert h == w

    return A.Compose(
        [
            A.SmallestMaxSize(h, interpolation=interpolation),
            A.CenterCrop(height=h, width=w),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            A.ToTensorV2(),
        ]
    )


def get_letterbox_processor(
    image_size: tuple[int, int],
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
    interpolation=cv2.INTER_AREA,
) -> ImageProcessor:
    h, w = image_size
    assert h == w

    return A.Compose(
        [
            A.LongestMaxSize(h, interpolation=interpolation),
            A.PadIfNeeded(min_height=h, min_width=w),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            A.ToTensorV2(),
        ]
    )


def get_image_processor_method(processor_type: str):
    if processor_type == "default":
        return get_default_processor

    elif processor_type == "letterbox":
        return get_letterbox_processor

    else:
        raise Exception("Unsupported processor name")


class DataProcessor:
    def __init__(
        self,
        image_processor: ImageProcessor,
        class2id: dict[str, int],
        device: str | torch.device,
    ):
        self.image_processor = image_processor
        self.class2id = class2id
        self.device = device

    def process(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        processed = dict()

        if "class" in data:
            label = [self.class2id[c] for c in data["class"]]
            label = torch.tensor(label).to(self.device)
            processed["label"] = label

        image = [load_image(p) for p in data["image_path"]]
        image = [self.image_processor(image=i)["image"] for i in image]
        image = torch.stack(image).to(self.device)
        processed["image"] = image

        return processed
