"""
Multilayer Perceptron (MLP) model for image classification.

References:
    - https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

from torch import Tensor
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple MLP model for image classification.
    """
    def __init__(
        self,
        in_dim=28*28,
        hidden=256,
        num_classes=10,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Forward pass for the model.

        Args:
            x (Tensor): Input tensor of shape (B, 1, 28, 28).

        Returns:
            Tensor: Output logits of shape (B, num_classes).
        """
        return self.net(x)
