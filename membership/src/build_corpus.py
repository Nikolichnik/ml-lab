"""
Build a synthetic corpus of sentences for language model training.
"""

import random

import re

import requests

from bs4 import BeautifulSoup

from common.constants import (
    NEWLINE,
    EXTENSION_TXT,
    DIR_CORPUS,
)
from common.exception import WebScrapingException

from util.arguments import parse_arguments
from util.io import write_resource_file

ARGUMENTS_MAP = {
    "url": (str, "URL to scrape content from (optional).", ""),
    "n": (int, "Number of sentences to generate.", 2000),
    "name-prefix": (str, "Corpus file name prefix.", "synthetic"),
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

def scrape_web_content(url: str) -> str:
    """
    Scrape text content from a web page.

    Args:
        url (str): The URL to scrape.

    Returns:
        str: The extracted text content.
    
    Raises:
        requests.RequestException: If the web request fails.
        Exception: If content extraction fails.
    """
    try:
        # Add headers to avoid being blocked by some websites
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text_elements = soup.find_all(["p", "article", "section", "div", "h1", "h2", "h3", "h4", "h5", "h6"])
        text_content = []

        for element in text_elements:
            text = element.get_text().strip()

            if text and len(text) > 10:  # Filter out very short text snippets
                text_content.append(text)

        if not text_content:
            # Fallback: get all text if no specific elements found
            text_content = [soup.get_text().strip()]

        full_text = " ".join(text_content)
        full_text = re.sub(r"\s+", " ", full_text)
        full_text = re.sub(r"\n+", "\n", full_text)

        return full_text.strip()

    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch content from {url}: {e}")
    except Exception as e:
        raise WebScrapingException(f"Failed to extract content from {url}: {e}") from e

def text_to_sentences(text: str) -> list[str]:
    """
    Convert text into individual sentences.

    Args:
        text (str): The input text.

    Returns:
        list[str]: List of sentences.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)
    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        # Filter out very short or very long sentences
        if 10 <= len(sentence) <= 300:
            cleaned_sentences.append(sentence + ".")

    return cleaned_sentences

def build_corpus(url: str = "", n: int = 2000) -> str:
    """
    Build a corpus of sentences from synthetic generation or web scraping.

    Args:
        n (int): The number of sentences to generate (for synthetic corpus).
        url (str): URL to scrape content from. If provided, scrapes instead of generating.

    Returns:
        str: The generated corpus as a single string.
    """
    if url.strip():
        # Scrape content from the provided URL
        print(f"Scraping content from: {url}")

        try:
            web_content = scrape_web_content(url)
            sentences = text_to_sentences(web_content)

            print(f"Extracted {len(sentences)} sentences from web content.")

            return NEWLINE.join(sentences)
        except WebScrapingException as e:
            print(f"Error scraping web content: {e}")
            print("Falling back to synthetic generation...")

    # Otherwise, generate synthetic sentences
    sentences = [make_sentence() for _ in range(n)]

    return NEWLINE.join(sentences)

# pylint: disable=invalid-name
if __name__ == "__main__":
    random.seed(7)

    args = parse_arguments(ARGUMENTS_MAP, "Builds a corpus from synthetic generation or web scraping.")

    name_prefix = args.name_prefix or "web" if args.url else "synthetic"
    name_suffix = f"_{args.n}" if not args.url else ""
    output_file = f"{name_prefix}{name_suffix}{EXTENSION_TXT}"

    corpus = build_corpus(url=args.url, n=args.n)

    write_resource_file(
        DIR_CORPUS,
        output_file,
        content=corpus,
    )

    print(f"Wrote {output_file} with {len(corpus.split(NEWLINE))} sentences.")
