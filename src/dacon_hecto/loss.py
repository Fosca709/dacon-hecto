import torch
import torch.nn as nn
import torch.nn.functional as F


# Gradient Boosting Cross Entropy (GBCE) loss from
# Fine-grained Recognition: Accounting for Subtle Differences between Similar Classes
# (https://arxiv.org/abs/1912.06842)
class GBCE(nn.Module):
    def __init__(self, k: int, label_smoothing: float = 0.0):
        super().__init__()
        self.k = k
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.clone()
        B = input.shape[0]
        input_truth = input[torch.arange(B), target].view(-1, 1)
        input[torch.arange(B), target] = float("-inf")
        input_topk = torch.topk(input, k=self.k, dim=-1).values
        input_selected = torch.concat([input_truth, input_topk], dim=-1)

        target_new = torch.zeros_like(target)
        return F.cross_entropy(input_selected, target_new, label_smoothing=self.label_smoothing)


def get_loss_fn(loss_name: str, label_smoothing: float = 0.0, top_k: int = 15, **kwargs):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_name == "gbce":
        return GBCE(k=top_k, label_smoothing=label_smoothing)

    else:
        raise Exception("Unsupported loss name")


# Contrastive loss from
# TransFG: A Transformer Architecture for Fine-grained Recognition
# (https://arxiv.org/abs/2103.07976)
class ContrastiveLoss(nn.Module):
    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha

    def forward(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # z: [B, N] matrix of representations
        # label: [B,]
        z_n = F.normalize(z, dim=-1)
        score = z_n @ z_n.T

        B = label.shape[0]
        label_expanded = label.unsqueeze(1).expand(B, B)
        pairwise_label = (label_expanded == label_expanded.T).float()

        positive_loss = pairwise_label * (1 - score)
        negative_loss = (1 - pairwise_label) * torch.clamp(score - self.alpha, min=0)
        loss = (positive_loss + negative_loss).mean()
        return loss
