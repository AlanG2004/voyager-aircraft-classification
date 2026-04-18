"""Binary aircraft classifier: ImageNet-pretrained ResNet-18 backbone
with a two-layer classification head. Backbone is fine-tuned end-to-end
in the final block only; earlier layers frozen to keep runs fast and
prevent overfitting on small subsets.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet18(weights=weights)

    # freeze all but layer4 and the classifier head
    for name, param in backbone.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return backbone


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_batchnorm(model: nn.Module) -> None:
    """Keep BatchNorm layers in eval mode during training.

    When backbone weights are frozen via requires_grad=False, BN layers still
    update running_mean and running_var during forward passes in train() mode.
    That silent drift hurts pretrained features. For short-horizon fine-tuning
    of a mostly-frozen backbone, we hold BN statistics fixed by forcing eval()
    on every BN module after model.train() is called.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
