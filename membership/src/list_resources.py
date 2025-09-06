"""
Lists all available corpora and checkpoints.
"""

from typing import List

from common.constants import DIR_CORPORA, DIR_CHECKPOINTS

from util.io import print_logo, get_resource_files


if __name__ == "__main__":
    print_logo()

    corpora: List[str] = get_resource_files(DIR_CORPORA)
    checkpoints: List[str] = get_resource_files(DIR_CHECKPOINTS)

    print("Available corpora:\n")

    for corpus in corpora:
        print(f" - {corpus.split('.')[0]}")

    print("\nAvailable checkpoints:\n")

    for checkpoint in checkpoints:
        print(f" - {checkpoint.split('.')[0]}")

    print()
