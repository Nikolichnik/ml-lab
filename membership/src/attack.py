"""
Membership inference attack using per-sequence loss thresholding.

References:
    - https://ieeexplore.ieee.org/document/9833649
"""
from datetime import datetime

from torch import Tensor, device, no_grad, load
from torch.nn import CrossEntropyLoss
from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

import numpy as np

import pandas as pd

from common.constants import (
    DEVICE_CUDA,
    DEVICE_CPU,
    DIR_CORPORA,
    DIR_CHECKPOINTS,
    DIR_RUNS,
    EXTENSION_TXT,
    EXTENSION_PT,
    EXTENSION_CSV,
    DATE_FORMAT,
    TIME_FORMAT,
    SPACER_DEFAULT,
)

from dataset.lm_dataset import LMDataset, collate_batch

from vocab.vocab import RestoredVocab

from lm.word_lstm import WordLSTM

from util.arguments import parse_arguments
from util.path import get_resource_path, ensure_dir
from util.io import read_resource_file, print_csv_table, save_plots


ARGUMENTS_MAP = {
    "checkpoint": (str, "Checkpoint to load.", f"synthetic_2000{SPACER_DEFAULT}100"),
    "corpus": (str, "Corpus to use. Optional, if not provided, attempt to resolve it from the checkpoint name will be made.", ""),
    "batch-size": (int, "Batch size for training.", 64),
    "input": (str, "Input text for evaluation. Optional. If not provided, a set of example sentences will be used.", ""),
}

DEVICE = DEVICE_CUDA if is_cuda_available() else DEVICE_CPU


# pylint: disable=redefined-outer-name, invalid-name
def per_sequence_loss(
    model: WordLSTM,
    batch: tuple[Tensor, Tensor, Tensor],
    device: device,
) -> np.ndarray:
    """
    Compute the per-sequence loss for a batch.

    Args:
        model (WordLSTM): The language model.
        batch (tuple[Tensor, Tensor, Tensor]): A batch of (inputs, targets, mask).
        device (device): The device to run the computation on.

    Returns:
        np.ndarray: The per-sequence losses as a numpy array.
    """
    xs, ys, mask = batch
    xs, ys, mask = xs.to(device), ys.to(device), mask.to(device)
    criterion = CrossEntropyLoss(ignore_index=0, reduction="none")

    with no_grad():
        logits, _ = model(xs)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), ys.reshape(B*T)).view(B, T)
        seq_loss = (loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return seq_loss.cpu().numpy()


def losses_for_dataset(
    model: WordLSTM,
    dataset: LMDataset,
    batch_size: int,
    device: device,
) -> np.ndarray:
    """
    Compute the losses for a dataset by batching.

    Args:
        model (WordLSTM): The language model.
        dataset (LMDataset): The dataset to compute losses for.
        batch_size (int): The batch size for DataLoader.
        device (device): The device to run the computation on.

    Returns:
        np.ndarray: The per-sequence losses for the entire dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    all_losses = []

    for batch in loader:
        all_losses.append(per_sequence_loss(model, batch, device))

    return np.concatenate(all_losses, axis=0)


def score_sentence(
    model: WordLSTM,
    vocab: RestoredVocab,
    sentence: str,
) -> float:
    """
    Compute the membership score for a single sentence.

    Args:
        model (WordLSTM): The language model.
        vocab (RestoredVocab): The vocabulary for tokenization.
        sentence (str): The input sentence.

    Returns:
        float: The membership score (negative loss).
    """
    dataset = LMDataset([sentence], vocab=vocab)
    xs, ys, mask = collate_batch([dataset[0]])

    return -per_sequence_loss(model, (xs, ys, mask), DEVICE)[0]


if __name__ == "__main__":
    args = parse_arguments(ARGUMENTS_MAP, "Performs membership inference attack on language models.")

    corpus_name = args.corpus or args.checkpoint.split(SPACER_DEFAULT)[0] if SPACER_DEFAULT in args.checkpoint else None

    if not corpus_name:
        raise ValueError("Corpus name could not be determined.")

    # Load corpus and split (train = members, held-out = non-members)
    lines = read_resource_file(DIR_CORPORA, f"{corpus_name}{EXTENSION_TXT}").splitlines()
    n = len(lines)
    n_train = int(0.7 * n)
    train_lines = lines[:n_train]
    held_lines  = lines[n_train:]

    # Restore model + vocab
    checkpoint_path = f"{get_resource_path(DIR_CHECKPOINTS)}/{args.checkpoint}{EXTENSION_PT}"
    checkpoint = load(checkpoint_path, map_location="cpu")
    vocab = RestoredVocab(checkpoint["vocab"])

    # Build datasets that use the restored vocab
    train_ds = LMDataset(train_lines, vocab=vocab)
    held_ds  = LMDataset(held_lines,  vocab=vocab)

    model = WordLSTM(vocab_size=len(vocab.itos)).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Compute per-sequence losses
    train_losses = losses_for_dataset(model, train_ds, args.batch_size, DEVICE)
    held_losses  = losses_for_dataset(model, held_ds,  args.batch_size, DEVICE)

    # Membership scores: lower loss ⇒ higher membership likelihood
    y_true = np.array([1]*len(train_losses) + [0] * len(held_losses))  # 1=member
    scores = -np.concatenate([train_losses, held_losses])
    mia_auc = roc_auc_score(y_true, scores)

    print(f"Membership Inference AUC (loss-threshold): {mia_auc:.3f} (1.0 perfect, 0.5 random)")

    # Sentences to score
    if args.input.strip():
        sentences = args.input.strip().split("|")
    else:
        # If no input provided, use some training and held-out examples + some random sentences.
        sentences = [
            train_lines[0] if train_lines else "Alice paints portraits in watercolor at dawn.",
            "Are you suggesting coconuts migrate?",
            held_lines[0]  if held_lines  else "Mallory composes melodies with strings on weekends.",
            "Just a flesh wound.",
            "Alice writes essays in watercolor at dawn.",
            "Nikola builds AI pipelines on GCP with privacy-first design.",
            "Carol designs landscapes in charcoal on weekends.",
            "Well, she turned me into a newt!",
            "Ni!",
        ]

    output = "sentence,score,normalized_score"
    results = []

    for sentence in sentences:
        score = score_sentence(
            model=model,
            vocab=vocab,
            sentence=sentence,
        )
        normalized_score = max(0, min(1, (score + 17) / 17))

        results.append(
            {
                "sentence": sentence,
                "score": round(score, 3),
                "normalized_score": round(normalized_score, 3)
            }
        )

    # Save output
    now = datetime.now()
    current_date = now.strftime(DATE_FORMAT)
    current_time = now.strftime(TIME_FORMAT)
    output_dir = f"{get_resource_path(DIR_RUNS)}/{current_date}/{current_time}{SPACER_DEFAULT}{args.checkpoint}"
    output_file = f"output{EXTENSION_CSV}"

    ensure_dir(output_dir)
    csv_path = f"{output_dir}/{output_file}"

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    print_csv_table(
        path=csv_path,
    )

    save_plots(
        train_losses=train_losses,
        held_losses=held_losses,
        y_true=y_true,
        scores=scores,
        output_dir=output_dir,
    )
