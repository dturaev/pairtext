import argparse
import logging


class StoreWithFlag(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


# Configure logging
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Simple sentence aligner for parallel text. "
            "Works well for machine translation."
        ),
        prog="pairtext",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional arguments: source and target files
    parser.add_argument("file1", help="Source text file")
    parser.add_argument("file2", help="Target text file")

    # Is text already split to sentences or do we want to split it?
    parser.add_argument(
        "-t",
        "--tokenized",
        action="store_true",
        help=(
            "Text is already tokenized (one sentence per line). "
            "Unless this option is set, the text will be split to "
            "sentences using wtpsplit."
        ),
    )

    # Score penalty for paths that lead to 1:n or n:1, because
    # the DP algorithm tends to prefer them over 1:1 alignments
    parser.add_argument(
        "-g",
        "--group-penalty",
        default=0.4,
        type=float,
        metavar="FLOAT",
        help=(
            "Score penalty for paths that lead to 1:n or n:1 "
            "groups. A higher penalty leads to more possible 1:1 "
            "misalignments, while a lower penalty leads to more "
            "possible 1:n or n:1 misalignments. You may want to try "
            "values between ~0.2 and ~0.4."
        ),
    )

    # Output aligned sentences for visual inspection
    parser.add_argument(
        "-o",
        "--output-file",
        # default="alignments.txt",
        # type=argparse.FileType("w"),
        type=str,
        metavar="FILE",
        help="The aligned sentences are written to this file",
    )

    # Gapped alignment options, for 1:0 and 0:1 alignments
    gapped_group = parser.add_argument_group(
        "Gapped alignment", "Options for performing gapped alignments"
    )

    # Perform gapped alignment (default: False)
    gapped_group.add_argument(
        "--gapped",
        action="store_true",
        help="Perform gapped alignment (identify 1:0 and 0:1 sentences)",
    )

    # Minimum score for valid translation (1:1, and concatenation of all for 1:n/n:1)
    gapped_group.add_argument(
        "-m",
        "--minscore-translation",
        default=0.5,
        type=float,
        metavar="FLOAT",
        action=StoreWithFlag,
        help=(
            "Minimum score for valid translation (1:1, 1:n or n:1); "
            "a sensible value is between ~0.3 for literary translations "
            "and ~0.7 for interviews and technical texts."
        ),
    )

    # Small preference for the correct translation in 1:n segments being
    # the first or the last sentence, or the concatenation of all
    gapped_group.add_argument(
        "--special-penalty",
        default=0.025,
        type=float,
        metavar="FLOAT",
        action=StoreWithFlag,
        help=(
            "This is a special penalty relevant for 1:n and n:1 "
            "segments. It reflects a small preference for the "
            "correct translation being a concatenation of all "
            "sentences, or the first or last sentence. "
            "There's probably no need to change it."
        ),
    )

    # Tiny preference for the fact that the correct translation in 1:n
    # segments is the concatenation of all (vs. first/last sentence)
    gapped_group.add_argument(
        "--concat-preference",
        default=0.01,
        type=float,
        metavar="FLOAT",
        action=StoreWithFlag,
        help=(
            "This parameter is related to `--special-penalty` "
            "and reflects a very small preference for the "
            "correct translation being a concatenation of all "
            "sentences, compared to the first or last sentence. "
            "It doesn't do much, and there's probably no need to change it."
        ),
    )

    # Parse and validate arguments
    args = parser.parse_args()
    check_args(parser, args)

    return args


def format_args(arguments):
    """Format and return the script's parameters as a string."""
    formatted_output = []
    for k, v in vars(arguments).items():

        if "explicit" in k:
            continue

        if not arguments.gapped:
            # These arguments are only relevant for gapped alignment
            if (
                k == "minscore_translation"
                or k == "special_penalty"
                or k == "concat_preference"
            ):
                continue

        # Check if the value is a file-like object
        if hasattr(v, "name") and hasattr(v, "mode"):
            value_repr = f"<{v.__class__.__name__} name='{v.name}' mode='{v.mode}'>"
        else:
            value_repr = repr(v)
        formatted_output.append(f"{k:21}: {value_repr}")
    return "\n  " + "\n  ".join(formatted_output)


def check_args(parser, arguments):
    if (
        getattr(arguments, "minscore_translation_explicit", False)
        or getattr(arguments, "special_penalty_explicit", False)
        or getattr(arguments, "concat_preference_explicit", False)
    ):
        if not arguments.gapped:
            parser.error(
                "--gapped must be set if --minscore-translation or "
                "--concat-preference or "
                "--special-penalty is used"
            )


def main():
    """Main entry point for the command line interface."""

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    args = parse_args()
    logger.info(format_args(args))

    # Import the implementation module only after parsing arguments
    from pairtext.core import run_alignment

    run_alignment(args)


if __name__ == "__main__":
    main()
