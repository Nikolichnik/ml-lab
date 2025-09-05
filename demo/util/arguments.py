"""
Argument parsing utilities.
"""

import argparse


def parse_arguments(argument_map: dict) -> argparse.Namespace:
    """
    Parse command-line arguments based on a provided argument map.
    Each key in the argument_map corresponds to an argument name, and its value is a tuple
    containing the type and default value for that argument.
    
    Args:
        argument_map (dict): A dictionary where keys are argument names and values are tuples of (type, default).

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        argument_map = {
            "epochs": (int, 10),
            "batch_size": (int, 32),
            "lr": (float, 0.001)
        }
        args = parse_arguments(argument_map)
    """
    parser = argparse.ArgumentParser()

    for arg, (arg_type, default) in argument_map.items():
        parser.add_argument(f"--{arg}", type=arg_type, default=default)

    return parser.parse_args()
