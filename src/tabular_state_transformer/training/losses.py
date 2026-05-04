from __future__ import annotations

import torch


def make_loss(task: str):
    return torch.nn.CrossEntropyLoss() if task == "classification" else torch.nn.MSELoss()
