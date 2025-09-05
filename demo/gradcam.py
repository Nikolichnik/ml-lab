"""
Simple demo of Grad-CAM on CIFAR-10 using SmallCNN and random weights.
"""

from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plot

from common.constants import DEVICE_CUDA, DEVICE_CPU, PATH_DATA

from model.cnn import SmallCNN

from util.gradcam import GradCAM


DEVICE = DEVICE_CUDA if is_cuda_available() else DEVICE_CPU

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds  = datasets.CIFAR10(root=PATH_DATA, train=False, download=True, transform=transform)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=True)

    model = SmallCNN().to(DEVICE)
    model.eval()  # using random weights unless you've trained/saved; still shows CAM plumbing

    target_layer = model.conv3
    cam_tool = GradCAM(model, target_layer)

    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(DEVICE)
    cam = cam_tool(imgs)
    cam = cam[0,0].cpu().numpy()

    plot.imshow(imgs[0].permute(1,2,0).cpu())
    plot.imshow(cam, alpha=0.5)
    plot.title("Grad-CAM (untrained weights)")
    plot.axis("off")
    plot.show()
