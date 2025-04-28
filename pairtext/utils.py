import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
# https://huggingface.co/sentence-transformers
# Loading the model takes time and memory, should happen only once
transformer_model: SentenceTransformer | None = None
transformer_model_name: str = ""


def compute_similarity_matrix(
    source_sentences: str | list[str],
    target_sentences: str | list[str],
    model: SentenceTransformer | str = "sentence-transformers/LaBSE",
) -> np.ndarray:
    """
    Compute cosine similarity matrix between two groups of sentences.
    """
    # Load sentence transformer model
    if isinstance(model, str):
        model = _load_model(model)

    # `model.encode` works with lists and strings, but `model.similarity_pairwise`
    # works only with 2D arrays/tensors, so we need to convert strings to lists
    if isinstance(source_sentences, str):
        source_sentences = [source_sentences]
    if isinstance(target_sentences, str):
        target_sentences = [target_sentences]

    # Encode sentences
    # model.encode(source_sentence, convert_to_numpy=True)
    src_embeddings = model.encode(source_sentences)
    tgt_embeddings = model.encode(target_sentences)

    # Calculate cosine similarity
    # similarity_matrix = np.dot(src_embeddings, tgt_embeddings.T)
    # similarity_matrix = model.similarity_pairwise(src_embeddings, tgt_embeddings)
    similarity_matrix = model.similarity(src_embeddings, tgt_embeddings)

    # Convert torch.Tensor to numpy array to make things easier
    return similarity_matrix.numpy()


def write_aligned_sentences_to_file(
    source_sentences: list[str],
    target_sentences: list[str],
    alignments: list[tuple[list[int], list[int]]],
    output_file: str,
):
    """
    Write aligned source and target sentences to a file.
    """
    with open(output_file, "w", encoding="utf-8") as fout:
        for source_indices, target_indices in alignments:
            fout.write("--------------------\n")
            # Write the source sentences joined by '||'
            fout.write("||".join([source_sentences[i] for i in source_indices]))
            fout.write("\n")

            # Write the target sentences joined by '||'
            fout.write("||".join([target_sentences[j] for j in target_indices]))
            fout.write("\n")


def _load_model(model_name: str = "sentence-transformers/LaBSE") -> SentenceTransformer:
    """Load the sentence transformer model once, to save time and memory."""
    global transformer_model
    global transformer_model_name

    # Load the model if not already loaded (for multiple models, we could
    # use module-level singleton class or lru_cache decorator)
    if transformer_model and model_name == transformer_model_name:
        return transformer_model

    try:
        transformer_model = SentenceTransformer(model_name)
        transformer_model_name = model_name
        return transformer_model
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
        raise  # Re-raise exception if model loading fails


def _concat_sentences(
    sentences: list[str], max_concat_length: int | None = None, bound: str | None = None
) -> tuple[list[str], list[list[int]]]:
    """Create all possible concatenations of successive sentences.

    >>> _concat_sentences(["a", "b", "c"])
    (['a b', 'b c', 'a b c', [[0, 1], [1, 2], [0, 1, 2]])
    >>> _concat_sentences(["a", "b", "c", "d"], bound="right")
    (['c d', 'b c d', 'a b c d'], [[2, 3], [1, 2, 3], [0, 1, 2, 3]])
    """
    if max_concat_length is None:
        max_concat_length = len(sentences)
    if bound == "both":
        bound = None

    sentences_concat = []
    indices = []
    for n_sents in range(1, max_concat_length):
        for i in range(len(sentences) - n_sents):

            if bound == "left" and i > 0:
                continue
            if bound == "right" and i < len(sentences) - n_sents - 1:
                continue

            concat = " ".join(sentences[i : i + n_sents + 1])
            sentences_concat.append(concat)
            indices.append([i for i in range(i, i + n_sents + 1)])

    return sentences_concat, indices
