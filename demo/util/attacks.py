"""
Adversarial Attacks: FGSM and PGD.

References:
    - https://arxiv.org/abs/1412.6572
    - https://arxiv.org/abs/1706.06083
"""

from torch import Tensor, clamp
from torch.nn import Module, functional as F

def fgsm(
    model: Module,
    x: Tensor,
    y: Tensor,
    eps: float,
) -> Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.

    Args:
        model (Module): The model to attack.
        x (Tensor): Input images.
        y (Tensor): True labels.
        eps (float): Perturbation magnitude.

    Returns:
        Tensor: Adversarial examples.
    """
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()

    return clamp(x_adv, 0, 1).detach()

def pgd(
    model: Module,
    x: Tensor,
    y: Tensor,
    eps: float = 0.03,
    alpha: float = 0.007,
    iters: int = 10
) -> Tensor:
    """
    Projected Gradient Descent (PGD) attack.

    Args:
        model (Module): The model to attack.
        x (Tensor): Input images.
        y (Tensor): True labels.
        eps (float): Maximum perturbation.
        alpha (float): Step size.
        iters (int): Number of iterations.

    Returns:
        Tensor: Adversarial examples.
    """
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    for _ in range(iters):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        eta = clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = clamp(x_orig + eta, 0, 1).detach()

    return x_adv
