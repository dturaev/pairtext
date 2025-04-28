import copy
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Iterator

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
_sentinel = object()

# Type Aliases for clarity
Indices = list[int]
AlignmentPair = tuple[Indices, Indices]
Alignment = list[AlignmentPair]
SentenceList = list[str]
Model = SentenceTransformer


class Bound(StrEnum):
    """Markers for the way how sentences are concatenated."""

    L = "left"
    R = "right"
    LR = "both"


class Marker(StrEnum):
    """Markers for splitting 1:n and n:1 alignment pairs."""

    L = "left"
    R = "right"


@dataclass
class AlignmentResult:
    """
    Represents the result of sentence alignment, containing
    the aligned sentence index pairs, associated scores, and
    references to the original sentence lists.
    """

    _pairs: Alignment
    src_sentences: SentenceList
    tgt_sentences: SentenceList
    matrix: np.ndarray
    model_name: str
    scores: list[list[float | None]] | None = None  # 1:1 scores
    scores_concat: list[float | None] | None = None  # concatenated sentences

    def __post_init__(self) -> None:
        if self.scores and len(self._pairs) != len(self.scores):
            raise ValueError("Length of '_pairs' and 'scores' must match.")
        self.validate_indices()

    def __len__(self) -> int:
        """Return number of alignment pairs."""
        return len(self._pairs)

    def __getitem__(self, index: int) -> AlignmentPair:
        """Get alignment pair tuple at index."""
        # `__getitem__` predates iterator protocol, but it's not going away
        # https://discuss.python.org/t/deprecate-old-style-iteration-protocol/17863
        return self._pairs[index]

    def __iter__(self) -> Iterator[AlignmentPair]:
        """Return an iterator over the alignment pairs."""
        # Without `__iter__`, Mypy complains
        return iter(self._pairs)

    @property
    def result(self) -> Alignment:
        return self._pairs

    def get_score(self, index: int) -> list[float | None] | None:
        """Get score for alignment pair at index."""
        # Only 1:1 scores (scores after first pass)
        if self.scores is None:
            return None
        if index >= len(self.scores):
            raise IndexError("Index out of bounds for scores list")
        return self.scores[index]

    def get_concat_score(self, index: int) -> float | None:
        """Get score for alignment pair at index."""
        # Also 1:n and n:1 scores (concatenated)
        if self.scores_concat is None:
            return None
        if index >= len(self.scores_concat):
            raise IndexError("Index out of bounds for scores list")
        return self.scores_concat[index]

    def get_pair_sentences(self, index: int) -> tuple[SentenceList, SentenceList]:
        """Get source/target sentences for alignment pair at given index."""
        src_indices, tgt_indices = self._pairs[index]
        src_sents = [self.src_sentences[i] for i in src_indices]
        tgt_sents = [self.tgt_sentences[i] for i in tgt_indices]
        return src_sents, tgt_sents

    def iter_pairs(self) -> Iterable[tuple[AlignmentPair, float | None]]:
        """
        Iterate through alignment pairs and their corresponding scores.
        Yield tuples of ((src_indices, tgt_indices), score).
        """
        if self.scores:
            if len(self._pairs) != len(self.scores):
                logger.warning(
                    "Mismatch between pairs and scores length during iteration."
                )
                scores_iter = self.scores + [None] * (
                    len(self._pairs) - len(self.scores)
                )
            else:
                scores_iter = self.scores
            yield from zip(self._pairs, scores_iter)
        else:
            for pair in self._pairs:
                yield pair, None

    def validate_indices(self) -> bool:
        """Check if alignment indices are valid."""
        max_src_idx = len(self.src_sentences) - 1
        max_tgt_idx = len(self.tgt_sentences) - 1
        for src_indices, tgt_indices in self._pairs:
            for idx in src_indices:
                if not (0 <= idx <= max_src_idx):
                    raise ValueError(f"Index {idx} > ({max_src_idx})")
            for idx in tgt_indices:
                if not (0 <= idx <= max_tgt_idx):
                    raise ValueError(f"Index {idx} > ({max_tgt_idx})")
        if self.scores and len(self._pairs) != len(self.scores):
            raise ValueError("Mismatch between '_pairs' and 'scores'.")
        if self.scores_concat and len(self._pairs) != len(self.scores_concat):
            raise ValueError("Mismatch between '_pairs' and 'scores_concat'.")
        return True

    @staticmethod
    def split_indices(
        indices: Indices, sub_indices: Indices, with_marker=False
    ) -> tuple[Indices, Indices, Indices]:
        """Split sentence indices based on a given subset."""
        # _subdivide(full_list, sub_list) returns (left_part, center_part, right_part)
        # based on the elements present in sub_list relative to full_list.
        left, center, right = _subdivide_list(indices, sub_indices)
        assert center
        if with_marker:
            left = [Marker.L] + left if left else []
            right = [Marker.R] + right if right else []
        return left, center, right

    def add_pair(
        self,
        src_indices: Indices,
        tgt_indices: Indices,
        score: float | None | object = _sentinel,
        score_concat: float | None | object = _sentinel,
    ):
        """Add new alignment pair"""
        # Modify in place or return new?
        self._pairs.append((src_indices, tgt_indices))
        if score is not _sentinel:
            if self.scores is None:
                self.scores = []
            self.scores.append([score])
        if score_concat is not _sentinel:
            if self.scores_concat is None:
                self.scores_concat = []
            self.scores_concat.append(score_concat)

    def update_pair(
        self,
        index: int,
        src_indices: Indices,
        tgt_indices: Indices,
        score: float | None | object = _sentinel,
        score_concat: float | None | object = _sentinel,
    ):
        """
        Update the alignment pair at the given index and optionally update its scores.

        Args:
            index: The index of the pair to update.
            src_indices: New source sentence indices.
            tgt_indices: New target sentence indices.
            score: New score; if not provided (sentinel), it remains unchanged.
            score_concat: New concatenated score, otherwise unchanged.
        """
        self._pairs[index] = (src_indices, tgt_indices)
        if score is not _sentinel:
            if not self.scores:
                raise ValueError("scores must be initialized before updating")
            self.scores[index] = [score]
        if score_concat is not _sentinel:
            if not self.scores_concat:
                raise ValueError("scores_concat must be initialized before updating")
            self.scores_concat[index] = score_concat

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        num_pairs = len(self)
        if self.scores:
            score_info = "with scores"  # 1:1 scores (won't work for n:m)
        elif self.scores_concat:
            score_info = "with scores_concat"  # 1:n scores
        else:
            score_info = "without scores"
        model_info = f" (model: {self.model_name})" if self.model_name else ""
        return (
            f"<AlignmentResult({num_pairs} pairs, "
            f"{len(self.src_sentences)} src, {len(self.tgt_sentences)} tgt, "
            f"{score_info}{model_info})>"
        )

    # Override the standard shallow copy mechanism
    # def __copy__(self):  # dunder method used by copy.copy
    def copy(self, drop_result: bool = False) -> "AlignmentResult":
        """
        Create a copy of an AlignmentResult instance.
        """
        new_pairs = [] if drop_result else copy.deepcopy(self._pairs)
        new_scores = None if drop_result else copy.deepcopy(self.scores)
        new_scores_concat = None if drop_result else copy.deepcopy(self.scores_concat)

        new_object = self.__class__(
            new_pairs,
            self.src_sentences,
            self.tgt_sentences,
            self.matrix,
            self.model_name,
            scores=new_scores,
            scores_concat=new_scores_concat,
        )
        return new_object

    # --- Other potential methods ---
    # def filter_by_score(self, min_score: float) -> 'AlignmentResult':
    #     """Return new AlignmentResult with pairs above a score threshold. """
    #     # Implementation would filter self._pairs and self.scores
    #     pass


# class ScoreProxy:
#     # The proxy class allows to provide array-like (i.e. getitem/setitem)
#     # semantics without cluttering the main class with index-specific
#     # methods.
#     # Useful for more complex behavior/interface, otherwise we can can
#     # define property methods directly in the class.
#     def __init__(self, alignment_result: "AlignmentResult") -> None:
#         self._alignment_result = alignment_result

#     def __getitem__(self, index: int) -> float | None:
#         return self._alignment_result.get_score(index)

#     def __setitem__(self, index: int, value: float) -> None:
#         if self._alignment_result.scores is None:
#             raise AttributeError("Scores are not initialized.")
#         if index >= len(self._alignment_result.scores):
#             raise IndexError("Index out of bounds for scores list")
#         self._alignment_result.scores[index] = value

#     def __len__(self) -> int:
#         if self._alignment_result.scores is None:
#             return 0
#         return len(self._alignment_result.scores)
#             return 0
#         return len(self._alignment_result.scores)


def _subdivide_list(full_list, sub_list):
    """
    Split a list of sorted indices into three parts: elements smaller than,
    equal to, and greater than the given subset.

    >>> _subdivide([4, 5, 6, 7, 8], [5, 6])
    ([4], [5, 6], [7, 8])
    >>> _subdivide([4, 5], [4, 5])
    ([], [4, 5], [])
    """
    assert not set(sub_list) - set(
        full_list
    ), f"{sub_list} must be subset of {full_list}"

    smaller = []
    equal = []
    larger = []
    sub_list.sort()

    for el in full_list:
        if el in sub_list:
            equal.append(el)
        elif el < sub_list[0]:
            smaller.append(el)
        else:
            assert el > sub_list[-1]
            larger.append(el)

    return smaller, equal, larger
