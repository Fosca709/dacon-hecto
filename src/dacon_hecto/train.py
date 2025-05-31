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

from .model import Classifier
from .process import DataProcessor

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
) -> None:
    if save_path is not None and os.path.exists(save_path):
        raise Exception("save_path already exists")

    if log_path is not None and os.path.exists(log_path):
        raise Exception("log_path already exists")

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epoch
    val_steps = len(val_dataloader)
    val_metrics = defaultdict(list)

    with Progress() as progress:
        train_bar = progress.add_task("train", total=total_steps)
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
                progress.update(train_bar, advance=1, refresh=True)

            if val_dataloader is not None:
                with torch.no_grad():
                    model.eval()
                    logits = []
                    labels = []

                    val_bar = progress.add_task("validation", total=val_steps)
                    for batch in val_dataloader:
                        batch = processor.process(batch)
                        image = batch["image"]
                        label = batch["label"]

                        output = model(image)

                        labels.append(label.detach().cpu())
                        logits.append(output.detach().cpu())

                        progress.update(val_bar, advance=1, refresh=True)

                    logits = torch.concat(logits)
                    labels = torch.concat(labels)

                    loss = F.cross_entropy(logits, labels).item()
                    preds = logits.argmax(dim=-1)
                    accuracy = ((preds == labels).sum() / labels.numel()).item()

                    tracker.run.log({"val/epoch": ep, "val/loss": loss, "val/accuracy": accuracy})
                    val_metrics["loss"].append(loss)
                    val_metrics["accuracy"].append(accuracy)

                    progress.update(val_bar, visible=False, refresh=True)

                    logger.info(f"epoch {ep}: loss {loss} accuracy {accuracy}")

                    if log_path is not None:
                        epoch_log_path = log_path / f"epoch_{ep}"
                        os.makedirs(epoch_log_path)
                        torch.save(logits, epoch_log_path / "logits.pt")
                        torch.save(labels, epoch_log_path / "labels.pt")

            if save_path is not None:
                model.save_model(save_path / f"epoch_{ep}")

    val_metrics = {k: np.array(v) for k, v in val_metrics.items()}
    best_epoch = np.argmin(val_metrics["loss"]).item()
    val_metrics = {k: v[best_epoch].item() for k, v in val_metrics.items()}
    val_metrics["best_epoch"] = best_epoch
    tracker.run.summary.update(val_metrics)

    tracker.run.finish()
