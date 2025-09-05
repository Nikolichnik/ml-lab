"""
Adversarial training demonstration using FGSM and PGD attacks.
"""

from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from common.constants import DEVICE_CUDA, DEVICE_CPU, PATH_DATA

from model.cnn import SmallCNN

from util.arguments import parse_arguments
from util.training import seed_everything, eval_epoch
from util.attacks import fgsm, pgd


DEVICE = DEVICE_CUDA if is_cuda_available() else DEVICE_CPU
ARGUMENTS_MAP = {
    "eps": (float, 0.03),
    "iters": (int, 10),
}

if __name__ == "__main__":
    args = parse_arguments(ARGUMENTS_MAP)

    seed_everything(42)

    transform = transforms.Compose([transforms.ToTensor()])
    test_ds  = datasets.CIFAR10(root=PATH_DATA, train=False, download=True, transform=transform)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = SmallCNN().to(DEVICE)
    criterion = CrossEntropyLoss()

    clean_loss, clean_acc, _, _ = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"Clean: acc={clean_acc:.3f}")

    x,y = next(iter(test_loader))
    x,y = x.to(DEVICE), y.to(DEVICE)
    x_fgsm = fgsm(model, x, y, eps=args.eps)
    x_pgd  = pgd(model, x, y, eps=args.eps, alpha=args.eps/4, iters=args.iters)

    with no_grad():
        acc_fgsm = (model(x_fgsm).argmax(1) == y).float().mean().item()
        acc_pgd  = (model(x_pgd).argmax(1) == y).float().mean().item()

    print(f"Adversarial (FGSM eps={args.eps}): acc={acc_fgsm:.3f}")
    print(f"Adversarial (PGD  eps={args.eps}, iters={args.iters}): acc={acc_pgd:.3f}")
