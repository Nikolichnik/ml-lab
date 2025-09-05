"""
Utility functions for training and evaluation.
"""

import random

import torch

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm

import numpy as np


def seed_everything(
    seed: int = 42,
) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(
    model: Module,
    loader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    device: str
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (Module): The model to train.
        loader (DataLoader): DataLoader for training data.
        criterion (Module): Loss function.
        optimizer (Optimizer): Optimizer for updating model weights.
        device (str): Device to run the training on ("cpu" or "cuda").

    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        predictions = logits.argmax(1)
        correct += (predictions == y).sum().item()
        total += y.size(0)

    return total_loss/total, correct/total

@torch.no_grad()
def eval_epoch(
    model: Module,
    loader: DataLoader,
    criterion: Module,
    device: str
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Evaluate the model for one epoch.

    Args:
        model (Module): The model to evaluate.
        loader (DataLoader): DataLoader for evaluation data.
        criterion (Module): Loss function.
        device (str): Device to run the evaluation on ("cpu" or "cuda").

    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_targets = [], []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())
        predictions = logits.argmax(1)
        correct += (predictions == y).sum().item()
        total += y.size(0)

    return total_loss/total, correct/total, torch.cat(all_probs), torch.cat(all_targets)
