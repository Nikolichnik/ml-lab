"""
Build a synthetic corpus of sentences for language model training.
"""

import random

from common.constants import (
    NEWLINE,
    EXTENSION_TXT,
    DIR_DATA,
)

from util.arguments import parse_arguments
from util.io import write_resource_file

ARGUMENTS_MAP = {
    "n": (int, "Number of sentences to generate.", 2000),
    "out": (str, "Output file name.", "synthetic_corpus"),
}

# pylint: disable=line-too-long
SUBJECTS = ["Alice", "Bob", "Carol", "Dave", "Eve", "Mallory", "Peggy", "Trent", "Victor", "Walter"]
VERBS = ["paints", "writes", "composes", "sketches", "records", "designs", "curates", "edits", "crafts", "imagines"]
OBJECTS = ["portraits", "stories", "poems", "melodies", "landscapes", "comics", "scenes", "lyrics", "haikus", "essays"]
STYLES = ["in watercolor", "in oil", "in charcoal", "with synths", "in pastel", "with ink", "in pencil", "with strings"]
CONTEXT = ["at dawn", "at night", "on weekends", "in spring", "by the river", "on stage", "in the studio", "in Vienna"]

def make_sentence():
    """
    Generate a random sentence.

    Returns:
        str: A randomly constructed sentence.
    """
    s = random.choice(SUBJECTS)
    v = random.choice(VERBS)
    o = random.choice(OBJECTS)
    st = random.choice(STYLES)
    c = random.choice(CONTEXT)

    return f"{s} {v} {o} {st} {c}."

def build_corpus(n:int = 2000) -> str:
    """
    Build a synthetic corpus of sentences.

    Args:
        n (int): The number of sentences to generate.

    Returns:
        str: The generated corpus as a single string.
    """
    sentences = [make_sentence() for _ in range(n)]

    return NEWLINE.join(sentences)

# pylint: disable=invalid-name
if __name__ == "__main__":
    args = parse_arguments(ARGUMENTS_MAP)

    random.seed(7)
    output_file = f"{args.out}_{args.n}{EXTENSION_TXT}"

    write_resource_file(
        DIR_DATA,
        output_file,
        content=build_corpus(args.n),
    )

    print(f"Wrote {output_file} with {args.n} sentences.")
