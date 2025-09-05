"""
A small Convolutional Neural Network (CNN) model for image classification.

References:
    - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    """
    A small CNN model for image classification.
    """
    def __init__(
        self,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32x32 -> 32x32
        self.pool = nn.MaxPool2d(2,2)                 # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 16x16 -> 16x16

        # After second pooling: 16x16 -> 8x8
        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass for the model.

        Args:
            x (Tensor): Input tensor of shape (B, 3, 32, 32).

        Returns:
            Tensor: Output logits of shape (B, num_classes).
        """
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # -> 16x16
        x = self.pool(F.relu(self.conv3(x)))  # -> 8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
