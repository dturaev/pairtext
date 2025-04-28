"""Tokenize text to sentences using wtpsplit."""

import logging
import os

from wtpsplit import SaT

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
# https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

# https://github.com/segment-any-text/wtpsplit?tab=readme-ov-file#usage
# https://arxiv.org/pdf/2406.16678
model = SaT("sat-3l-sm")
# model = SaT("sat-6l-sm")

chars_to_strip = {
    '"',  # Various quote marks
    "“",
    "”",
    "„",
    "‘",
    "’",
    "'",
    "`",
    "´",
    "«",
    "»",
    # ")",  # Closing brackets/parentheses - may be relevant
    # "]",
    # "}",
    # ">",
    "》",
    "」",
    "』",
    # ",",  # Other punctuation that shouldn't end a sentence
    # ";",  # may be relevant
    # ":",  # may be relevant
    # " ",  # Whitespaces (we're using `strip()` to clean them out)
    # "\t",
    # "\n",
}

# Those symbols can have a strong impact on sentence similarity. For example:
# >>> compute_similarity_matrix('“Powerful warm, warn’t it?”', '„Sehr heiß, he?“')
# 0.72659194
# >>> compute_similarity_matrix('Powerful warm, warn’t it?', 'Sehr heiß, he?')
# 0.6219296
# Another example:
# >>> s = '“Smarty!'
# >>> t = '„Pah — das kann jeder sagen!“'
# >>> compute_similarity_matrix(s, t)
# 0.5158747
# >>> s = 'Smarty!'
# >>> t = 'Pah — das kann jeder sagen!'
# >>> compute_similarity_matrix(s, t)
# 0.36890447

# Standard sentence endings
sentence_endings = {".", "!", "?", "…", "。", "！", "？"}


def clean_sentence(sentence):
    """Clean a sentence by removing unwanted characters."""
    cleaned = sentence.rstrip()
    if any(cleaned.endswith(end) for end in sentence_endings):
        return cleaned

    # If no valid ending yet, strip unwanted characters
    while cleaned and any(cleaned.endswith(char) for char in chars_to_strip):
        for char in chars_to_strip:
            if cleaned.endswith(char):
                cleaned = cleaned.rstrip(char)
                break

    # Add period if no sentence ending is present
    # to make sentences more comparable (?)
    if cleaned and not any(cleaned.endswith(end) for end in sentence_endings):
        cleaned += "."

    return cleaned


def sentence_tokenize(text, model=model, clean=False):
    """Tokenize text to sentences using wtpsplit."""
    logger.debug("Splitting text to sentences...")
    results = [s.strip() for s in model.split(text) if s.strip()]

    if clean:
        results_new = []
        for sentence in results:
            sentence = clean_sentence(sentence.strip())
            if sentence:
                results_new.append(sentence)
        results = results_new

    return results
