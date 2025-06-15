import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from rich.progress import track
from torch.utils.data import DataLoader

from .config import BaseConfig
from .data import (
    get_class_names,
    get_dataloader,
    get_debug_dataframes,
    get_test_dataframe,
    get_train_dataframe,
    overlapped_categories,
    train_val_split,
)
from .model import Classifier
from .process import DataProcessor, TestDataProcessor, get_image_processor_method

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <bold>{message}</bold>",
    colorize=True,
)


def _eval(
    model: Classifier, dataloader: DataLoader, processor: DataProcessor, refresh_per_second: float = 1.0
) -> torch.Tensor:
    model.eval()

    logits = []
    with torch.no_grad():
        for batch in track(dataloader, description="Evaluation", refresh_per_second=refresh_per_second):
            batch = processor.process_eval(batch)
            image = batch["image"]

            output = model(image)
            logits.append(output.cpu())

    logits = torch.concat(logits)
    return logits


def get_probs(logits: torch.Tensor, num_data_per_image: int = 1):
    probs = torch.softmax(logits, dim=-1)

    if num_data_per_image == 1:
        return probs

    probs = probs.reshape(-1, num_data_per_image, logits.shape[-1])
    probs = probs.mean(dim=1)
    return probs


def smoothing(probs: torch.Tensor, epsilon: float) -> torch.Tensor:
    probs = probs.clone()
    num_labels = probs.shape[-1]

    probs += epsilon

    preds = probs.argmax(dim=-1)
    probs[torch.arange(probs.shape[0]), preds] -= num_labels * epsilon
    return probs


class EvalConfig(BaseConfig):
    mode: Literal["submission", "validation", "debug"] = "submission"
    num_data_per_batch: int = 32
    device: str = "cuda"
    processor_type: str = "letterbox"
    use_tta: bool = False
    tta_n: int | None = None
    tta_r: int | None = None
    smoothing: float = 0.0


def eval(
    model: Classifier,
    config: EvalConfig,
    data_path: Path,
    submission_path: Path | None = None,
    refresh_per_second: float = 1.0,
) -> None:
    class_names = get_class_names(data_path)
    class2id = {c: i for i, c in enumerate(class_names)}

    mode = config.mode
    if mode == "submission":
        if submission_path is None:
            raise Exception("`submission_path` must be set in submission mode")
        df = get_test_dataframe(data_path)

    elif mode == "validation":
        df = get_train_dataframe(data_path)
        _, df = train_val_split(df)

    elif mode == "debug":
        df = get_train_dataframe(data_path)
        _, df = get_debug_dataframes(df)

    else:
        raise Exception("Unsupported eval mode")

    dataloader = get_dataloader(df, num_data_per_batch=config.num_data_per_batch, shuffle=False)

    image_processor_method = get_image_processor_method(config.processor_type)
    image_processor = image_processor_method(
        image_size=model.config.image_size,
        normalize_mean=model.config.normalize_mean,
        normalize_std=model.config.normalize_std,
    )

    processor = TestDataProcessor(
        image_processor=image_processor,
        class2id=class2id,
        device=config.device,
        use_tta=config.use_tta,
        n=config.tta_n,
        r=config.tta_r,
    )

    if config.use_tta:
        logger.info("Use Test Time Augmentation")
    logger.info(f"num_data_per_image: {processor.num_data_per_image}")
    logger.info(f"batch size: {config.num_data_per_batch * processor.num_data_per_image}")

    logits = _eval(model=model, dataloader=dataloader, processor=processor, refresh_per_second=refresh_per_second)
    probs = get_probs(logits, num_data_per_image=processor.num_data_per_image)

    if config.smoothing > 0.0:
        probs = smoothing(probs, epsilon=config.smoothing)

    if mode != "submission":
        labels = df["class"].replace_strict(class2id).to_torch()
        loss = F.nll_loss(torch.log(probs), labels)
        logger.info(f"loss: {loss.item()}")
        return

    submission = make_submission(probs=probs.numpy(), ids=df["ID"], class_names=class_names)
    submission.write_csv(submission_path / "submission.csv")


def make_submission(probs: np.ndarray, ids: pl.Series, class_names: list[str]) -> pl.DataFrame:
    df = pl.DataFrame(probs, schema=class_names)
    df = df.select(ids, pl.all(), *[pl.lit(0.0, dtype=pl.Float32).alias(name) for name in overlapped_categories.keys()])
    return df


def _make_confusion_pairs(probs: torch.Tensor, num_pairs: int, count_limit: int, verbose: bool = True):
    top2 = torch.topk(probs, k=2)
    top2_probs = top2.values
    top2_prob_diffs = (top2_probs[:, 0] - top2_probs[:, 1]).tolist()
    top2_labels = top2.indices.tolist()

    prob_diffs = defaultdict(float)
    counter = defaultdict(int)
    for d, ls in zip(top2_prob_diffs, top2_labels):
        ls = tuple(sorted(ls))
        prob_diffs[ls] += d
        counter[ls] += 1

    prob_means = {k: (prob_diffs[k] / v) for k, v in counter.items() if v >= count_limit}
    pairs = sorted(prob_means, key=lambda x: prob_means[x])

    if len(pairs) < num_pairs:
        raise Exception(
            f"Not enough eligible pairs (count_limit={count_limit}). Max available: {len(pairs)}, requested: {num_pairs}."
        )

    pairs = pairs[:num_pairs]
    num_data = sum([counter[k] for k in pairs])

    if verbose:
        logger.info(f"Eligible pairs (count ≥ {count_limit}): {len(prob_means)}")
        logger.info(f"Selected data covers {num_data / probs.size(0):.2%} of the dataset")

    return pairs


def make_confusion_pairs(
    model: Classifier,
    config: EvalConfig,
    num_pairs: int,
    count_limit: int,
    data_path: Path,
    save_name: str = "confusion_pairs.pkl",
    refresh_per_second: float = 1.0,
) -> None:
    class_names = get_class_names(data_path)
    class2id = {c: i for i, c in enumerate(class_names)}
    id2class = {i: c for i, c in enumerate(class_names)}

    mode = config.mode
    if mode == "submission":
        df = get_test_dataframe(data_path)

    elif mode == "validation":
        df = get_train_dataframe(data_path)
        _, df = train_val_split(df)

    elif mode == "debug":
        df = get_train_dataframe(data_path)
        _, df = get_debug_dataframes(df)

    else:
        raise Exception("Unsupported eval mode")

    if config.use_tta:
        logger.warning("Note: `use_tta=True` is set, but this method does not use TTA.")

    dataloader = get_dataloader(df, num_data_per_batch=config.num_data_per_batch, shuffle=False)

    image_processor_method = get_image_processor_method(config.processor_type)
    image_processor = image_processor_method(
        image_size=model.config.image_size,
        normalize_mean=model.config.normalize_mean,
        normalize_std=model.config.normalize_std,
    )
    processor = DataProcessor(image_processor=image_processor, class2id=class2id, device=config.device)

    logits = _eval(model=model, dataloader=dataloader, processor=processor, refresh_per_second=refresh_per_second)
    probs = get_probs(logits=logits, num_data_per_image=processor.num_data_per_image)

    pairs = _make_confusion_pairs(probs=probs, num_pairs=num_pairs, count_limit=count_limit, verbose=True)
    pairs = [(id2class[a], id2class[b]) for a, b in pairs]

    with open(data_path / save_name, "wb") as f:
        pickle.dump(pairs, f)


def ensemble(submissions: list[pl.DataFrame], class_names: list[str], epsilon: float = 0.0) -> pl.DataFrame:
    values = []
    ids = submissions[0]["ID"]

    for df in submissions:
        cols = df.columns[1:-5]
        assert cols == class_names

        values.append(df[cols].to_numpy())

    probs = sum(values) / len(submissions)

    if epsilon > 0.0:
        preds = np.argmax(probs, axis=1)
        probs = np.clip(probs, a_min=epsilon, a_max=1.0)
        probs[np.arange(probs.shape[0]), preds] = 0.0
        pred_probs = 1 - probs.sum(axis=1)
        probs[np.arange(probs.shape[0]), preds] = pred_probs

    return make_submission(probs, ids, class_names)
