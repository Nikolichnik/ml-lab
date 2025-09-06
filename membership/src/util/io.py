"""
Utility functions for working with I/O.
"""

import pathlib

from sklearn.metrics import roc_curve, auc

import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from common.constants import ENCODING_UTF8, DIR_LOGOS, LOGO_DEFAULT

from util.path import get_resource_path, ensure_dir


matplotlib.use("Agg")


def get_resource_files(
    *children: str,
) -> list[str]:
    """
    Get all file names in a resource directory.

    Args:
        *children (str): Subdirectories to append to the base resource path.

    Returns:
        list[str]: Sorted list of file names in the directory.
    """
    dir_path = get_resource_path(*children)
    p = pathlib.Path(dir_path)

    if not p.exists() or not p.is_dir():
        return []

    return sorted([f.name for f in p.iterdir() if f.is_file()])

def read_resource_file(
    *children: str,
    encoding: str = ENCODING_UTF8,
) -> str:
    """
    Read the contents of a resource file.

    Args:
        *children (str): Subdirectories or file names to append to the base resource path.
        encoding (str): The encoding to use when reading the file.

    Returns:
        str: The contents of the file.
    """
    file_path = get_resource_path(*children)

    return pathlib.Path(file_path).read_text(encoding=encoding)


def write_resource_file(
    *children: str,
    content: str,
    encoding: str = ENCODING_UTF8,
) -> None:
    """
    Write content to a resource file, creating directories as needed.

    Args:
        *children (str): Subdirectories or file names to append to the base resource path.
        content (str): The content to write to the file.
        encoding (str): The encoding to use when writing the file.

    Returns:
        None
    """
    file_path = get_resource_path(*children)
    dir_path = str(pathlib.Path(file_path).parent)

    ensure_dir(dir_path)

    pathlib.Path(file_path).write_text(content, encoding=encoding)


def print_logo(variant: str = LOGO_DEFAULT) -> None:
    """
    Print the logo by reading its content from the resource directory.

    Args:
        variant (str): The logo to print.
    """
    logo_content = read_resource_file(DIR_LOGOS, variant)

    print(f"\n{logo_content}\n")


def print_table(
    headers: list[str],
    rows: list[list[str]],
    padding: int = 1,
    max_column_width: int = 80,
) -> None:
    """
    Print a formatted table to the console.

    Args:
        headers (list[str]): The table headers.
        rows (list[list[str]]): The table rows.
        padding (int): Number of blank lines to print before and after the table.
        max_column_width (int): Maximum width of a table column.

    Returns:
        None
    """
    def truncate_text(text: str, max_width: int) -> str:
        """
        Truncate text to max_width, adding '...' if needed.

        Args:
            text (str): The text to truncate.
            max_width (int): The maximum allowed width.

        Returns:
            str: The truncated text.
        """
        if len(text) <= max_width:
            return text

        return text[:max_width - 3] + "..."

    # Ensure all rows have the same number of columns as headers
    num_columns = len(headers)
    normalized_rows = []
    for row in rows:
        # Pad row with empty strings if it's too short, or truncate if too long
        normalized_row = row[:num_columns] + [""] * (num_columns - len(row))
        normalized_rows.append(normalized_row)

    # Truncate all content first
    truncated_headers = [truncate_text(header, max_column_width) for header in headers]
    truncated_rows = [
        [truncate_text(str(item), max_column_width) for item in row]
        for row in normalized_rows
    ]

    # Calculate column widths based on truncated content
    col_widths = [
        max(len(str(item)) for item in col) 
        for col in zip(*([truncated_headers] + truncated_rows))
    ]

    header_line = " │ ".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(truncated_headers))
    separator_line = "─┼─".join("─" * col_width for col_width in col_widths)

    for _ in range(padding):
        print()

    print(header_line)
    print(separator_line)

    for row in truncated_rows:
        row_line = " │ ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row))
        print(row_line)

    for _ in range(padding):
        print()


def print_csv_table(
    path: str,
    separator: str = ",",
    max_column_width: int = 80,
    percentage_columns: list[str] = None,
    highlight_threshold: float = 70.0,
) -> None:
    """
    Print a CSV file as a formatted table to the console using pandas.

    Args:
        path (str): The path to the CSV file.
        separator (str): The delimiter used in the CSV file.
        max_column_width (int): Maximum width of a table column.
        percentage_columns (list[str]): Column names to format as percentages for display.
        highlight_threshold (float): Threshold for highlighting percentage values.

    Returns:
        None
    """
    # ANSI color codes
    BOLD = "\033[1m"
    RED = "\033[91m"
    RESET = "\033[0m"

    def highlight_percentage(value: float, threshold: float) -> str:
        """
        Apply color highlighting to percentage values above threshold.

        Args:
            value (float): The percentage value.
            threshold (float): The threshold for highlighting.

        Returns:
            str: The formatted (and possibly highlighted) percentage string.
        """
        formatted_value = f"{value:.1f}"

        if value >= threshold:
            return f"{BOLD}{RED}{formatted_value}{RESET}"

        return formatted_value

    try:
        # Read CSV with pandas
        df = pd.read_csv(get_resource_path(path), sep=separator)

        if df.empty:
            print("No data to display.")

            return

        # Create a copy for display formatting (don't modify original)
        display_df = df.copy()

        # Format specified columns as percentages
        if percentage_columns:
            for col in percentage_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: highlight_percentage(x * 100, highlight_threshold)
                    )

        # Convert to the format expected by print_table
        headers = []

        for col in display_df.columns:
            header = str(col).strip().capitalize().replace("_", " ")

            # Add [%] suffix to percentage column headers
            if percentage_columns and col in percentage_columns:
                header += " [%]"

            headers.append(header)

        rows = [
            [str(item) for item in row]
            for row in display_df.values
        ]

        print_table(headers, rows, max_column_width=max_column_width)

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")


def save_plots(
    train_losses: np.ndarray,
    held_losses: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    output_dir: str,
) -> None:
    """
    Save plots for loss distributions and ROC curve.

    Args:
        train_losses (np.ndarray): Losses for training (member) data.
        held_losses (np.ndarray): Losses for held-out (non-member) data.
        y_true (np.ndarray): True membership labels.
        scores (np.ndarray): Membership scores.
        output_path (str): Directory to save the plots.

    Returns:
        None
    """

    plot_dir = f"{output_dir}/plot"
    ensure_dir(plot_dir)
    plot_dir_name = f"{output_dir.split('/')[-1]}/plot"

    # Plot histograms of losses
    plt.figure(figsize=(6,4))
    plt.hist(train_losses, bins=30, alpha=0.6, label="Members (train)")
    plt.hist(held_losses, bins=30, alpha=0.6, label="Non-members (held-out)")
    plt.xlabel("Per-sequence loss")
    plt.ylabel("Count")
    plt.title("Membership inference: loss distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/loss_histogram.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Membership inference ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/roc_curve.png")
    plt.close()

    print(f"Histogram and ROC plots available in {plot_dir_name}.")
