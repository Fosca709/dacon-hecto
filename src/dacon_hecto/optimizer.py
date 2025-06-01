import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.optimization import get_constant_schedule_with_warmup


def get_optimizer(
    model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float, **kwargs
) -> Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)

    else:
        raise Exception("Unsupported optimizer name")


def get_scheduler(
    optimizer: Optimizer, scheduler_name: str, training_steps: int, warmup_steps: int, **kwargs
) -> LRScheduler:
    if scheduler_name == "constant":
        return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, **kwargs)

    else:
        raise Exception("Unsupported scheduler name")
