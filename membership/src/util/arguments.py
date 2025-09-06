"""
Argument parsing utilities.
"""

import argparse

from util.io import print_logo


def parse_arguments(argument_map: dict, description: str = None) -> argparse.Namespace:
    """
    Parse command-line arguments based on a provided argument map.
    Each key in the argument_map corresponds to an argument name, and its value is a tuple
    containing the type and default value for that argument.
    
    Args:
        argument_map (dict): A dictionary where keys are argument names and values are tuples of (type, default).
        description (str): Optional description for the argument parser.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        argument_map = {
            "epochs": (int, 10),
            "batch_size": (int, 32),
            "lr": (float, 0.001)
        }
        args = parse_arguments(argument_map, "Train a neural network")
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False  # Disable default help to customize it
    )

    # Add custom help argument
    parser.add_argument(
        "--help", "-h",
        action="help",
        help="Show this help message and exit."
    )

    # Track used short options to avoid conflicts
    used_short_options = {'h'}  # help is already used

    for arg, (arg_type, arg_description, default) in argument_map.items():
        # Generate unique short option
        short_option = None
        for char in arg:
            if char.lower() not in used_short_options:
                short_option = f"-{char.lower()}"
                used_short_options.add(char.lower())
                break

        # Add argument with or without short option
        if short_option:
            parser.add_argument(
                f"--{arg}", short_option,
                type=arg_type, 
                default=default,
                metavar="",
                help=f"[{arg_type.__name__}] {arg_description}"
            )
        else:
            parser.add_argument(
                f"--{arg}",
                type=arg_type,
                default=default,
                metavar="",
                help=f"[{arg_type.__name__}] {arg_description}"
            )

    try:
        print_logo()
        args = parser.parse_args()

        return args
    except SystemExit as e:
        # Always print help when argparse exits
        if e.code != 0:
            print("\nInvalid arguments. See usage below:\n")
            parser.print_help()

        raise e
