"""
Train a language model on a text corpus.
"""

from torch import Tensor, save
from torch.cuda import is_available as is_cuda_available
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from common.constants import (
    DEVICE_CUDA,
    DEVICE_CPU,
    KEY_MODEL,
    KEY_VOCAB,
    DIR_CORPUS,
    DIR_CHECKPOINTS,
    EXTENSION_TXT,
    EXTENSION_PT,
    SPACER_DEFAULT,
)

from util.arguments import parse_arguments
from util.path import get_resource_path
from util.io import read_resource_file

from dataset.lm_dataset import LMDataset, collate_batch

from lm.word_lstm import WordLSTM


ARGUMENTS_MAP = {
    "corpus": (str, "Corpus to use.", "synthetic_2000"),
    "epochs": (int, "Number of training epochs.", 100),
    "batch-size": (int, "Batch size for training.", 64),
    "learning-rate": (float, "Learning rate.", 2e-3),
    "name-prefix": (str, "Model checkpoint name prefix.", "")
}

DEVICE = DEVICE_CUDA if is_cuda_available() else DEVICE_CPU

# pylint: disable=redefined-outer-name, invalid-name
def loss_for_batch(
    logits: Tensor,
    y: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Compute the loss for a batch of sequences.

    Args:
        logits (Tensor): The model's output logits of shape (B, T, V).
        y (Tensor): The target token indices of shape (B, T).
        mask (Tensor): The mask indicating valid tokens of shape (B, T).

    Returns:
        Tensor: The average loss over the batch.
    """
    B,T,V = logits.shape
    criterion = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = criterion(logits.reshape(B*T, V), y.reshape(B*T))
    loss = loss.view(B,T)
    seq_loss = (loss*mask).sum(dim=1)/(mask.sum(dim=1)+1e-8)

    return seq_loss.mean()

if __name__ == "__main__":
    args = parse_arguments(ARGUMENTS_MAP, "Trains a language model on a text corpus.")

    print(f"Training LM on {args.corpus} for {args.epochs} epochs using {DEVICE}.")

    text = read_resource_file(DIR_CORPUS, f"{args.corpus}{EXTENSION_TXT}").splitlines()
    n = len(text)
    n_train = int(0.7 * n)
    train_lines = text[:n_train]
    heldout_lines = text[n_train:]

    train_ds = LMDataset(train_lines)
    heldout_ds = LMDataset(heldout_lines, vocab=train_ds.vocab)

    vocab = train_ds.vocab
    model = WordLSTM(vocab_size=len(vocab.itos)).to(DEVICE)

    opt = Adam(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        steps = 0

        for xs, ys, mask in train_loader:
            xs, ys, mask = xs.to(DEVICE), ys.to(DEVICE), mask.to(DEVICE)
            logits, _ = model(xs)
            loss = loss_for_batch(logits, ys, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1

        print(f"Epoch {epoch}: train_loss={total/max(1, steps):.4f}")

    name_prefix = f"{args.name_prefix}{SPACER_DEFAULT}" if args.name_prefix else ""
    output_path = f"{get_resource_path(DIR_CHECKPOINTS)}/{name_prefix}{args.corpus}{SPACER_DEFAULT}{args.epochs}{EXTENSION_PT}"

    save(
        obj={
            KEY_MODEL: model.state_dict(),
            KEY_VOCAB: vocab.itos,
        },
        f=output_path,
    )

    print(f"LM saved to {output_path}")
