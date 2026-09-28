import torch
from timm.loss import SoftTargetCrossEntropy
from torch import nn
from torch.nn import functional as F

from .registry import lookup


class SoftmaxFocal(nn.Module):
    def __init__(self, gamma, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - logp.exp()).pow(self.gamma) * logp
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


class SigmoidFocal(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = logits.sigmoid()
        pt = p * target + (1 - p) * (1 - target)
        loss = loss * (1 - pt).pow(self.gamma)
        if self.alpha is not None:
            loss = loss * (self.alpha * target + (1 - self.alpha) * (1 - target))
        return loss.mean()


def build_loss(cfg, data):
    spec = cfg.loss
    weight = spec.class_weight
    if weight == "balanced":
        counts = list(data.report["class_counts"]["train"].values())
        if min(counts) == 0:
            raise ValueError("Balanced loss weights require all classes in train")
        weight = [sum(counts) / (len(counts) * x) for x in counts]
        spec.class_weight = weight
    if weight is not None:
        if len(weight) != len(data.classes) or min(weight) <= 0:
            raise ValueError("class_weight must contain one positive weight per class")
        weight = torch.tensor(weight, dtype=torch.float32)
    if spec.name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=spec.label_smoothing)
    if spec.name == "soft_target_ce":
        return SoftTargetCrossEntropy()
    if spec.name == "bce_with_logits":
        outputs = 1 if cfg.task.type == "binary" else len(data.classes)
        if spec.pos_weight is not None and (len(spec.pos_weight) != outputs or min(spec.pos_weight) <= 0):
            raise ValueError("pos_weight must contain one positive value per output")
        pos = torch.tensor(spec.pos_weight) if spec.pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pos)
    if spec.name == "softmax_focal":
        if spec.alpha is not None:
            raise ValueError("softmax_focal uses class_weight; scalar alpha is only for sigmoid_focal")
        return SoftmaxFocal(spec.gamma, weight)
    if spec.name == "sigmoid_focal":
        return SigmoidFocal(spec.gamma, spec.alpha)
    return lookup("loss", spec.name)(cfg=cfg, data=data)
