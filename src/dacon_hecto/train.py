import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from loguru import logger
from rich.progress import Progress
from torch.utils.data import DataLoader

from .config import BaseConfig
from .data import get_class_names, get_dataloader_from_config
from .loss import ContrastiveLoss, get_loss_fn
from .model import Classifier
from .optimizer import get_optimizer, get_scheduler
from .process import DataProcessor, DataProcessorWithAugmentation, get_image_processor_method
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


def _train(
    model: Classifier,
    processor: DataProcessor,
    # optimizer: Optimizer,
    # scheduler: LRScheduler,
    optimizer_name: str,
    lr_head: float,
    lr_head_full: float,
    lr_encoder: float,
    use_sam: bool,
    warmup_ratio: float,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    epoch: int,
    epoch_freeze: int,
    tracker: WandbTracker,
    loss_fn: nn.Module,
    use_contrastive: bool,
    contrastive_alpha: float,
    accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    save_path: Path | None = None,
    log_path: Path | None = None,
    hf_hub_manager: HFHubManager | None = None,
    progress_refresh_per_second: float = 1.0,
) -> None:
    assert epoch_freeze <= epoch

    if save_path is not None and os.path.exists(save_path):
        raise Exception("save_path already exists")

    if log_path is not None and os.path.exists(log_path):
        raise Exception("log_path already exists")

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epoch
    val_steps = len(val_dataloader)
    val_metrics = defaultdict(list)

    optimizer_head = get_optimizer(model.fc, optimizer_name=optimizer_name, learning_rate=lr_head, use_sam=use_sam)

    optimizer_encoder = get_optimizer(
        model.vision_encoder, optimizer_name=optimizer_name, learning_rate=lr_encoder, use_sam=use_sam
    )

    if use_contrastive:
        if use_sam:
            raise Exception("SAM was built without contrastive loss. I didn't update the code and don't plan to.")
        contrastive_loss_fn = ContrastiveLoss(alpha=contrastive_alpha)

    is_freeze = True
    model.set_encoder_grad(False)

    with Progress(refresh_per_second=progress_refresh_per_second) as progress:
        train_bar = progress.add_task("train", total=total_steps)
        val_bar = progress.add_task("validation", total=val_steps, visible=False)
        global_step = 1

        for ep in range(epoch):
            if is_freeze and (ep >= epoch_freeze):
                model.set_encoder_grad(True)
                is_freeze = False

                optimizer_head = get_optimizer(
                    model.fc, optimizer_name=optimizer_name, learning_rate=lr_head_full, use_sam=use_sam
                )

                training_steps = (epoch - epoch_freeze) * steps_per_epoch // accumulation_steps
                warmup_steps = int(training_steps * warmup_ratio)
                base_optim_head = optimizer_head.base_optimizer if use_sam else optimizer_head
                base_optim_encoder = optimizer_encoder.base_optimizer if use_sam else optimizer_encoder

                scheduler_head = get_scheduler(
                    base_optim_head, scheduler_name="constant", training_steps=training_steps, warmup_steps=warmup_steps
                )
                scheduler_encoder = get_scheduler(
                    base_optim_encoder,
                    scheduler_name="constant",
                    training_steps=training_steps,
                    warmup_steps=warmup_steps,
                )

            model.train()
            input_list = []
            for batch in train_dataloader:
                batch = processor.process_train(batch)
                image = batch["image"]
                label = batch["label"]
                if use_sam:
                    input_list.append(batch)

                z = model.vision_encoder(image)
                output = model.fc(z)

                loss = loss_fn(output, label)
                loss_info = {"train/step": global_step, "train/loss": loss.item()}

                if use_contrastive:
                    loss_contrastive = contrastive_loss_fn(z, label)
                    loss_info.update({"train/loss_contrastive": loss_contrastive.item()})
                    loss += loss_contrastive

                tracker.run.log(loss_info)
                (loss / accumulation_steps).backward()

                if tracker.is_param_update(step=global_step):
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            tracker.run.log({"param/step": global_step, f"param/{name}_norm": param.data.norm().item()})

                            if param.grad is not None:
                                tracker.run.log(
                                    {"grad/step": global_step, f"grad/{name}_norm": param.grad.norm().item()}
                                )

                if global_step % accumulation_steps == 0:
                    lr_info = {"train/step": global_step, "train/lr_head": optimizer_head.param_groups[0]["lr"]}

                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                    if use_sam:
                        optimizer_head.first_step(zero_grad=True)
                        if not is_freeze:
                            lr_info.update({"train/lr_encoder": optimizer_encoder.param_groups[0]["lr"]})
                            optimizer_encoder.first_step(zero_grad=True)

                        tracker.run.log(lr_info)

                        for batch in input_list:
                            image = batch["image"]
                            label = batch["label"]

                            output = model(image)
                            loss = loss_fn(output, label)
                            (loss / accumulation_steps).backward()

                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                        optimizer_head.second_step(zero_grad=True)
                        if not is_freeze:
                            optimizer_encoder.second_step(zero_grad=True)
                            scheduler_head.step()
                            scheduler_encoder.step()

                        input_list.clear()

                    else:
                        optimizer_head.step()

                        if not is_freeze:
                            lr_info.update({"train/lr_encoder": optimizer_encoder.param_groups[0]["lr"]})
                            optimizer_encoder.step()

                            scheduler_head.step()
                            scheduler_encoder.step()

                            optimizer_encoder.zero_grad()

                        tracker.run.log(lr_info)

                        optimizer_head.zero_grad()

                global_step += 1
                progress.update(train_bar, advance=1)

            if val_dataloader is not None:
                with torch.no_grad():
                    model.eval()
                    logits = []
                    labels = []

                    progress.update(val_bar, visible=True)
                    for batch in val_dataloader:
                        batch = processor.process_eval(batch)
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
    accumulation_steps: int = 1
    eval_batch_size: int = 32
    processor_type: str = "letterbox"
    use_augmentation: bool = True
    num_data_per_image: int = 4
    device: str = "cuda"
    optimizer: str = "adamw"
    lr_head: float = 1e-2
    lr_head_full: float = 1e-3
    lr_encoder: float = 1e-4
    use_sam: bool = False
    warmup_ratio: float = 0.1
    max_grad_norm: float | None = None
    epoch: int = 3
    epoch_freeze: int = 3
    loss: str = "cross_entropy"
    label_smoothing: float = 0.0
    gbce_top_k: int = 15
    use_contrastive: bool = False
    contrastive_alpha: float = 0.4
    use_wandb: bool = False
    project_name: str = "dacon-hecto"
    param_log_freq: int = -1
    push_to_hub: bool = False


def train(
    model: Classifier,
    config: TrainConfig,
    data_path: Path,
    save_path: Path | None = None,
    log_path: Path | None = None,
    progress_refresh_per_second: float = 1.0,
) -> None:
    class_names = get_class_names(data_path)
    class2id = {c: i for i, c in enumerate(class_names)}

    train_dataloader, val_dataloader = get_dataloader_from_config(
        data_path=data_path,
        num_data_per_batch=config.num_data_per_batch,
        eval_batch_size=config.eval_batch_size,
        use_val=config.use_val,
        debug=config.debug,
    )
    logger.info(f"train_dataloader size: {len(train_dataloader)}")
    size_val_dataloader = None if (val_dataloader is None) else len(val_dataloader)
    logger.info(f"val_dataloader size: {size_val_dataloader}")

    logger.info(f"image_size: {model.config.image_size}")
    image_processor_method = get_image_processor_method(processor_type=config.processor_type)
    image_processor = image_processor_method(
        image_size=model.config.image_size,
        normalize_mean=model.config.normalize_mean,
        normalize_std=model.config.normalize_std,
    )

    if config.use_augmentation:
        processor = DataProcessorWithAugmentation(
            image_processor=image_processor,
            class2id=class2id,
            device=config.device,
            num_data_per_image=config.num_data_per_image,
        )

    else:
        processor = DataProcessor(image_processor=image_processor, class2id=class2id, device=config.device)

    batch_size = config.num_data_per_batch * processor.num_data_per_image
    logger.info(f"batch size: {batch_size} / effective batch size: {batch_size * config.accumulation_steps}")

    loss_fn = get_loss_fn(loss_name=config.loss, label_smoothing=config.label_smoothing, top_k=config.gbce_top_k)

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
        optimizer_name=config.optimizer,
        lr_head=config.lr_head,
        lr_head_full=config.lr_head_full,
        lr_encoder=config.lr_encoder,
        use_sam=config.use_sam,
        warmup_ratio=config.warmup_ratio,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epoch=config.epoch,
        epoch_freeze=config.epoch_freeze,
        tracker=tracker,
        loss_fn=loss_fn,
        use_contrastive=config.use_contrastive,
        contrastive_alpha=config.contrastive_alpha,
        accumulation_steps=config.accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        save_path=save_path,
        log_path=log_path,
        hf_hub_manager=hf_hub_manager,
        progress_refresh_per_second=progress_refresh_per_second,
    )
