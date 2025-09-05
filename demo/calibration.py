"""
Calibration demo for evaluating model uncertainty.
"""

from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from torchvision import datasets, transforms

from common.constants import DEVICE_CUDA, DEVICE_CPU, PATH_DATA

from model.cnn import SmallCNN

from util.training import eval_epoch
from util.metrics import expected_calibration_error, plot_reliability_diagram


DEVICE = DEVICE_CUDA if is_cuda_available() else DEVICE_CPU

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds  = datasets.CIFAR10(root=PATH_DATA, train=False, download=True, transform=transform)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = SmallCNN().to(DEVICE)
    criterion = CrossEntropyLoss()

    te_loss, te_acc, te_probs, te_targets = eval_epoch(model, test_loader, criterion, DEVICE)
    probs = te_probs.numpy()
    targets = te_targets.numpy()
    ece, bins, bin_acc, bin_conf = expected_calibration_error(probs, targets, n_bins=15)

    print(f"Test acc={te_acc:.3f}  ECE={ece:.3f}")

    plot_reliability_diagram(
        bins=bins,
        bin_acc=bin_acc,
        bin_conf=bin_conf,
        title="Reliability Diagram (untrained model demo)"
    )
