import random
from itertools import combinations, product
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.augmentations.crops.functional import crop


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
                CropDown(p=0.25, ratio=(0.1, 0.4)),
                CropRight(p=0.25, ratio=(0.1, 0.4)),
                CropLeftAndUp(p=0.5, left=(0, 20), up=(0, 20)),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(p=0.5),
                A.Affine(p=0.5, rotate=(-10, 10)),
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
            "crop_x": (CropFixedPixel(p=1, left=5), CropFixedPixel(p=1, right=5)),
            "crop_y": (CropFixedPixel(p=1, down=5), CropFixedPixel(p=1, up=5)),
            "rotate": (A.Affine(rotate=5, p=1), A.Affine(rotate=-5, p=1)),
            "flip": (A.HorizontalFlip(p=1),),
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


def crop_pixel(img: np.ndarray, left: int = 0, right: int = 0, up: int = 0, down: int = 0) -> np.ndarray:
    h, w, _ = img.shape
    return crop(img=img, x_min=left, y_min=up, x_max=w - right, y_max=h - down)


def crop_ratio(
    img: np.ndarray, left: float = 0.0, right: float = 0.0, up: float = 0.0, down: float = 0.0
) -> np.ndarray:
    h, w, _ = img.shape
    x_min = int(w * left)
    x_max = int(w * (1 - right))
    y_min = int(h * up)
    y_max = int(h * (1 - down))
    return crop(img=img, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


class CropFixedPixel(A.ImageOnlyTransform):
    def __init__(self, p: float, left: int = 0, right: int = 0, up: int = 0, down: int = 0):
        super().__init__(p=p)
        self.left = left
        self.right = right
        self.up = up
        self.down = down

    def apply(self, img: np.ndarray, **params):
        return crop_pixel(img=img, left=self.left, right=self.right, up=self.up, down=self.down)


class CropLeftAndUp(A.ImageOnlyTransform):
    def __init__(self, p: float, left: tuple[int, int] = (0, 20), up: tuple[int, int] = (0, 20)):
        super().__init__(p=p)
        self.left = left
        self.up = up

    def apply(self, img: np.ndarray, **params):
        left_px = random.randint(self.left[0], self.left[1])
        up_px = random.randint(self.up[0], self.up[1])

        return crop_pixel(img=img, left=left_px, up=up_px)


class CropRight(A.ImageOnlyTransform):
    def __init__(self, p: float, ratio: tuple[float, float] = (0.0, 0.3)):
        super().__init__(p=p)
        self.ratio = ratio

    def apply(self, img: np.ndarray, **params):
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        return crop_ratio(img=img, right=ratio)


class CropDown(A.ImageOnlyTransform):
    def __init__(self, p: float, ratio: tuple[float, float] = (0.0, 0.3)):
        super().__init__(p=p)
        self.ratio = ratio

    def apply(self, img: np.ndarray, **params):
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        return crop_ratio(img=img, down=ratio)
