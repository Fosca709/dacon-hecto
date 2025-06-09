from itertools import combinations, product
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch


def load_image(image_path) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_default_processor(
    image_size: tuple[int, int],
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
    interpolation=cv2.INTER_AREA,
):
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
):
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
    def __init__(self, image_processor, class2id: dict[str, int], device: str | torch.device, **kwargs):
        self.image_processor = image_processor
        self.class2id = class2id
        self.device = device
        self.num_data_per_image = 1

    def process_eval(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
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

    def process_train(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.process_eval(data)


class DataProcessorWithAugmentation(DataProcessor):
    def __init__(
        self, image_processor, class2id: dict[str, int], device: str | torch.device, num_data_per_image: int, **kwargs
    ):
        super().__init__(image_processor=image_processor, class2id=class2id, device=device)
        self.num_data_per_image = num_data_per_image
        self.augment_processor = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_px=(-10, 10), rotate=(-10, 10), p=0.5),
                A.Blur(blur_limit=(3, 5), p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(p=0.5),
                self.image_processor,
            ]
        )

    def process_train(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        processed = dict()

        if "class" in data:
            label = [self.class2id[c] for c in data["class"]]
            label = [x for x in label for _ in range(self.num_data_per_image)]
            label = torch.tensor(label).to(self.device)
            processed["label"] = label

        image = [load_image(p) for p in data["image_path"]]
        image = [i for i in image for _ in range(self.num_data_per_image)]
        image = [self.augment_processor(image=i)["image"] for i in image]
        image = torch.stack(image).to(self.device)
        processed["image"] = image

        return processed


def group_combinations(groups: dict[str, Any], r: int):
    for group_keys in combinations(groups.keys(), r):
        for picks in product(*(groups[k] for k in group_keys)):
            yield picks


class TestDataProcessor(DataProcessor):
    def __init__(
        self,
        image_processor,
        class2id: dict[str, int],
        device: str | torch.device,
        use_tta: bool = False,
        n: int | None = None,
        r: int | None = None,
    ):
        super().__init__(image_processor=image_processor, class2id=class2id, device=device)
        self.transforms = {
            "flip": (A.HorizontalFlip(p=1),),
            "shift_x": (A.Affine(translate_px={"x": 5}, p=1), A.Affine(translate_px={"x": -5}, p=1)),
            "shift_y": (A.Affine(translate_px={"y": 5}, p=1), A.Affine(translate_px={"y": -5}, p=1)),
            "rotate": (A.Affine(rotate=5, p=1), A.Affine(rotate=-5, p=1)),
        }
        self.use_tta = use_tta
        self.n = n
        self.r = r

        self.__init_processor__()

    def __init_processor__(self) -> None:
        if not self.use_tta:
            self.num_data_per_image = 1
            self.processors = [self.image_processor]
            return

        if (self.n is None) == (self.r is None):
            raise Exception("Exactly one of `n` or `r` must be set (not both or neither).")

        r = len(self.transforms) if self.r is None else self.r
        assert r <= len(self.transforms)

        processors = [self.image_processor]
        count = 1
        for i in range(1, r + 1):
            iterator = group_combinations(self.transforms, r=i)
            for ts in iterator:
                if count == self.n:
                    break

                processors.append(A.Compose([*ts, self.image_processor]))
                count += 1

        if (self.n is not None) and (count != self.n):
            raise Exception(f"`n` is too large — the maximum allowed is {count}.")

        self.processors = processors
        self.num_data_per_image = len(processors)

    def process_eval(self, data):
        processed = dict()

        if "class" in data:
            label = [self.class2id[c] for c in data["class"]]
            label = [x for x in label for _ in range(self.num_data_per_image)]
            label = torch.tensor(label).to(self.device)
            processed["label"] = label

        image = [load_image(p) for p in data["image_path"]]
        image = [p(image=i)["image"] for i in image for p in self.processors]
        image = torch.stack(image).to(self.device)
        processed["image"] = image

        return processed
