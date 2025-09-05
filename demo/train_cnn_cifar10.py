"""
Train a small CNN on CIFAR-10 dataset and evaluate its performance.
"""

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from common.constants import DEVICE_CUDA, DEVICE_CPU, PATH_DATA

from model.cnn import SmallCNN

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

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = datasets.CIFAR10(root=PATH_DATA, train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root=PATH_DATA, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = SmallCNN().to(DEVICE)
    opt = Adam(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, opt, DEVICE)
        te_loss, te_acc, te_probs, te_targets = eval_epoch(model, test_loader, criterion, DEVICE)

        print(f"E{epoch}: train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")
