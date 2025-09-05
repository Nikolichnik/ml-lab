"""
Train a simple MLP on MNIST dataset.
"""

from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from torchvision import datasets, transforms

from common.constants import DEVICE_CUDA, DEVICE_CPU, PATH_DATA

from model.mlp import MLP

from util.arguments import parse_arguments
from util.training import seed_everything, train_epoch, eval_epoch


DEVICE = DEVICE_CUDA if is_cuda_available() else DEVICE_CPU
ARGUMENTS_MAP = {
    "epochs": (int, 2),
    "batch_size": (int, 128),
    "lr": (float, 1e-3)
}

if __name__ == "__main__":
    args = parse_arguments(ARGUMENTS_MAP)

    seed_everything(42)

    train_ds = datasets.MNIST(root=PATH_DATA, train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root=PATH_DATA, train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = MLP().to(DEVICE)
    opt = Adam(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, opt, DEVICE)
        te_loss, te_acc, te_probs, te_targets = eval_epoch(model, test_loader, criterion, DEVICE)

        print(f"E{epoch}: train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")
