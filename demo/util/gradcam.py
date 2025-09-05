"""
Grad-CAM implementation for visual explanations of CNN decisions.

References:
    - https://arxiv.org/abs/1610.02391
"""
from torch import Tensor
from torch.nn import Module, functional as F

class GradCAM:
    """
    Grad-CAM implementation for visual explanations of CNN decisions.
    """
    def __init__(
        self,
        model: Module,
        target_layer: Module,
    ):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    # pylint: disable=unused-argument
    def _register_hooks(self) -> None:
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(
        self,
        x: Tensor,
        class_idx: int | None = None,
    ) -> Tensor:
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(1)

        loss = logits[range(logits.size(0)), class_idx].sum()
        loss.backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)  # GAP over H,W
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def close(self) -> None:
        """
        Close the Grad-CAM hooks.
        """
        for h in self.hook_handles:
            h.remove()
