import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from rich.progress import Progress
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .config import BaseConfig
from .data import get_class_names, get_dataloader_from_config
from .model import Classifier
from .optimizer import get_optimizer, get_scheduler
from .process import DataProcessor, get_image_processor_method
from .utils import HFHubManager

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <bold>{message}</bold>",
    colorize=True,
)


class WandbTracker:
    def __init__(
        self,
        key: str,
        project: str,
        run_name: str,
        config,
        param_log_freq: int = -1,
    ):
        self.param_log_freq = param_log_freq

        wandb.login(key=key)
        self.run = wandb.init(project=project, name=run_name, config=config, settings=wandb.Settings(quiet=True))

        self.run.define_metric(name="train/*", step_metric="train/step")
        self.run.define_metric(name="val/*", step_metric="val/epoch")
        self.run.define_metric(name="param/*", step_metric="param/step")
        self.run.define_metric(name="grad/*", step_metric="grad/step")

    def is_param_update(self, step: int):
        if self.param_log_freq == -1:
            return False

        if step % self.param_log_freq == 0:
            return True

        return False


class MockTracker:
    def __init__(self, *args, **kwargs):
        self.run = MockRun()

    def is_param_update(self, step: int):
        return False


class MockRun:
    def __init__(self):
        self.summary = dict()

    def log(self, *args, **kwargs): ...

    def finish(self): ...


def get_loss_fn(loss_name: str):
    if loss_name == "cross_entropy":
        return F.cross_entropy
    else:
        raise Exception("Unsupported loss name")


def _train(
    model: Classifier,
    processor: DataProcessor,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    epoch: int,
    tracker: WandbTracker,
    loss_fn=F.cross_entropy,
    save_path: Path | None = None,
    log_path: Path | None = None,
    hf_hub_manager: HFHubManager | None = None,
) -> None:
    if save_path is not None and os.path.exists(save_path):
        raise Exception("save_path already exists")

    if log_path is not None and os.path.exists(log_path):
        raise Exception("log_path already exists")

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epoch
    val_steps = len(val_dataloader)
    val_metrics = defaultdict(list)

    with Progress(refresh_per_second=1.0) as progress:
        train_bar = progress.add_task("train", total=total_steps)
        val_bar = progress.add_task("validation", total=val_steps, visible=False)
        global_step = 1

        for ep in range(epoch):
            model.train()
            for batch in train_dataloader:
                batch = processor.process(batch)
                image = batch["image"]
                label = batch["label"]

                output = model(image)
                loss = loss_fn(output, label)
                loss.backward()

                tracker.run.log(
                    {"train/step": global_step, "train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]}
                )

                if tracker.is_param_update(step=global_step):
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            tracker.run.log({"param/step": global_step, f"param/{name}_norm": param.data.norm().item()})

                            if param.grad is not None:
                                tracker.run.log(
                                    {"grad/step": global_step, f"grad/{name}_norm": param.grad.norm().item()}
                                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                progress.update(train_bar, advance=1)

            if val_dataloader is not None:
                with torch.no_grad():
                    model.eval()
                    logits = []
                    labels = []

                    progress.update(val_bar, visible=True)
                    for batch in val_dataloader:
                        batch = processor.process(batch)
                        image = batch["image"]
                        label = batch["label"]

                        output = model(image)

                        labels.append(label.detach().cpu())
                        logits.append(output.detach().cpu())

                        progress.update(val_bar, advance=1)

                    logits = torch.concat(logits)
                    labels = torch.concat(labels)

                    loss = F.cross_entropy(logits, labels).item()
                    preds = logits.argmax(dim=-1)
                    accuracy = ((preds == labels).sum() / labels.numel()).item()

                    tracker.run.log({"val/epoch": ep, "val/loss": loss, "val/accuracy": accuracy})
                    val_metrics["loss"].append(loss)
                    val_metrics["accuracy"].append(accuracy)

                    progress.reset(val_bar, total=val_steps, visible=False)

                    logger.info(f"epoch {ep}: loss {loss} accuracy {accuracy}")

                    if log_path is not None:
                        epoch_log_path = log_path / f"epoch_{ep}"
                        os.makedirs(epoch_log_path)
                        torch.save(logits, epoch_log_path / "logits.pt")
                        torch.save(labels, epoch_log_path / "labels.pt")

            if save_path is not None:
                model.save_model(save_path / f"epoch_{ep}")

    if val_dataloader is not None:
        val_metrics = {k: np.array(v) for k, v in val_metrics.items()}
        best_epoch = np.argmin(val_metrics["loss"]).item()
        val_metrics = {k: v[best_epoch].item() for k, v in val_metrics.items()}
        val_metrics["best_epoch"] = best_epoch
        tracker.run.summary.update(val_metrics)

    tracker.run.finish()

    if hf_hub_manager is not None:
        if save_path is None:
            logger.warning("`hf_hub_manager` requires `save_path` to be set.")

        elif val_dataloader is None:
            logger.warning("`hf_hub_manager` currently doesn't work without a `val_dataloader`.")

        else:
            save_name = save_path.name
            folder_path = save_path / f"epoch_{best_epoch}"
            hf_hub_manager.push_to_hub(folder_path=folder_path, save_name=save_name)


class TrainConfig(BaseConfig):
    run_name: str = "test"
    use_val: bool = True
    debug: bool = False
    num_data_per_batch: int = 32
    processor_type: str = "default"
    device: str = "cuda"
    optimizer: str = "adamw"
    learning_rate: float = 1e-2
    weight_dacay: float = 0.01
    scheduler: str = "constant"
    warmup_ratio: float = 0.1
    epoch: int = 3
    loss: str = "cross_entropy"
    use_wandb: bool = False
    project_name: str = "dacon-hecto"
    param_log_freq: int = -1
    push_to_hub: bool = False


def train(
    model: Classifier, config: TrainConfig, data_path: Path, save_path: Path | None = None, log_path: Path | None = None
) -> None:
    class_names = get_class_names(data_path)
    class2id = {c: i for i, c in enumerate(class_names)}

    train_dataloader, val_dataloader = get_dataloader_from_config(
        data_path=data_path, num_data_per_batch=config.num_data_per_batch, use_val=config.use_val, debug=config.debug
    )
    logger.info(f"train_dataloader size: {len(train_dataloader)}")
    size_val_dataloader = None if (val_dataloader is None) else len(val_dataloader)
    logger.info(f"val_dataloader size: {size_val_dataloader}")

    # TODO: This should be modified after data augmentation has been implemented.
    logger.info(f"effective batch size: {config.num_data_per_batch}")

    logger.info(f"image_size: {model.config.image_size}")
    image_processor_method = get_image_processor_method(processor_type=config.processor_type)
    image_processor = image_processor_method(
        image_size=model.config.image_size,
        normalize_mean=model.config.normalize_mean,
        normalize_std=model.config.normalize_std,
    )
    processor = DataProcessor(image_processor=image_processor, class2id=class2id, device=config.device)

    optimizer = get_optimizer(
        model=model,
        optimizer_name=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_dacay,
    )

    training_steps = len(train_dataloader) * config.epoch
    warmup_steps = int(training_steps * config.warmup_ratio)
    scheduler = get_scheduler(
        optimizer=optimizer, scheduler_name=config.scheduler, training_steps=training_steps, warmup_steps=warmup_steps
    )

    loss_fn = get_loss_fn(config.loss)

    if config.use_wandb:
        try:
            WANDB_API_KEY = os.environ["WANDB_API_KEY"]

        except KeyError:
            raise Exception("The environment variable `WANDB_API_KEY` must be set to use wandb.")

        project_name = config.project_name
        if config.debug:
            project_name += "-debug"

        config_dict = config.__to_dict__()
        config_dict["model_config"] = model.config.__to_dict__()

        tracker = WandbTracker(
            key=WANDB_API_KEY,
            project=project_name,
            run_name=config.run_name,
            config=config_dict,
            param_log_freq=config.param_log_freq,
        )

    else:
        tracker = MockTracker()

    if save_path is None:
        logger.info("Model saving is disabled")

    else:
        save_path = save_path / config.run_name

    if log_path is None:
        logger.info("Log saving is disabled")

    else:
        log_path = log_path / config.run_name

    if config.push_to_hub:
        try:
            HF_API_TOKEN = os.environ["HF_API_TOKEN"]
            HF_REPO_ID = os.environ["HF_REPO_ID"]
        except KeyError:
            raise Exception("The environment variables `HF_API_KEY` and `HF_REPO_ID` must be set to use `push_to_hub`")

        hf_hub_manager = HFHubManager(token=HF_API_TOKEN, repo_id=HF_REPO_ID)

    else:
        logger.info("Skipping model upload to HuggingFace Hub.")
        hf_hub_manager = None

    _train(
        model=model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epoch=config.epoch,
        tracker=tracker,
        loss_fn=loss_fn,
        save_path=save_path,
        log_path=log_path,
        hf_hub_manager=hf_hub_manager,
    )
