import logging

import numpy as np

from pairtext.datatypes import Alignment, AlignmentResult
from pairtext.utils import compute_similarity_matrix

logger = logging.getLogger(__name__)


def align_sentences(
    source_sentences: list[str],
    target_sentences: list[str],
    group_penalty: float = 0.4,
    model_name: str = "sentence-transformers/LaBSE",
) -> AlignmentResult:
    """
    Align source and target sentences using cosine similarity.

    Returns a list of source-target sentence mappings.
    Each mapping is a tuple of two lists of indices. The first list contains
    the indices of the source sentences, and the second list contains the
    indices of the aligned target sentences.
    """
    # Compute similarity matrix between source and target sentences
    similarity_matrix = compute_similarity_matrix(
        source_sentences, target_sentences, model_name
    )

    # Find alignment (optimal path through the similarity matrix)
    max_score, alignment_path = max_path(similarity_matrix, penalty=group_penalty)
    best_scores = [similarity_matrix[r, c] for r, c in alignment_path]

    # Group consecutive alignments into human-readable tuple format
    alignments: Alignment = _group_consecutive_alignments(alignment_path)

    # Group scores by consecutive alignments
    scores_grouped: list[list[float]] = []
    offset = 0
    for a in alignments:
        n_sent = max(len(a[0]), len(a[1]))
        scores_grouped.append([best_scores[i + offset] for i in range(n_sent)])
        offset += n_sent

    for source_indices, target_indices in alignments:
        logger.debug(
            f"--------------------\n"
            f"{'||'.join([source_sentences[i] for i in source_indices])}\n"
            f"{'||'.join([target_sentences[j] for j in target_indices])}"
        )

    alignment = AlignmentResult(
        alignments,
        src_sentences=source_sentences,
        tgt_sentences=target_sentences,
        matrix=similarity_matrix,
        model_name=model_name,
        scores=scores_grouped,
    )

    return alignment


def max_path(matrix, penalty=0.3):
    """
    Find the maximum score path through a similarity matrix
    using dynamic programming (similar to Needleman-Wunsch
    with affine gap penalties).
    """
    num_rows, num_cols = matrix.shape

    # Initialize DP table with -infinity
    dp = np.full((num_rows, num_cols, 3), -np.inf)
    # Backtracking pointers: each entry holds (prev_row, prev_col, prev_direction)
    backtrack = np.zeros((num_rows, num_cols, 3), dtype=object)

    # Base case: starting position (0,0)
    dp[0, 0, 0] = matrix[0, 0]  # R (right)
    dp[0, 0, 1] = matrix[0, 0]  # D (down)
    dp[0, 0, 2] = matrix[0, 0]  # Diag (diagonal)

    for row in range(num_rows):
        for col in range(num_cols):
            current_val = matrix[row, col]

            # Update Right move (0)
            if col >= 1:
                # Can come from previous Right (0) or Diagonal (2)
                prev_right = dp[row, col - 1, 0]
                prev_diag_right = dp[row, col - 1, 2]
                max_val = max(prev_right, prev_diag_right)
                if max_val != -np.inf:
                    # Penalty prevents that algorithm maximizes score
                    # by breaking up more sentences than necessary
                    dp[row, col, 0] = max_val + current_val - penalty

                    if prev_right >= prev_diag_right:
                        backtrack[row, col, 0] = (row, col - 1, 0)
                    else:
                        backtrack[row, col, 0] = (row, col - 1, 2)

            # Update Down move (1)
            if row >= 1:
                # Can come from previous Down (1) or Diagonal (2)
                prev_down = dp[row - 1, col, 1]
                prev_diag_down = dp[row - 1, col, 2]
                max_val = max(prev_down, prev_diag_down)
                if max_val != -np.inf:
                    dp[row, col, 1] = max_val + current_val - penalty

                    if prev_down >= prev_diag_down:
                        backtrack[row, col, 1] = (row - 1, col, 1)
                    else:
                        backtrack[row, col, 1] = (row - 1, col, 2)

            # Update Diagonal move (2)
            if row >= 1 and col >= 1:
                # Can come from any previous direction
                prev_right_diag = dp[row - 1, col - 1, 0]
                prev_down_diag = dp[row - 1, col - 1, 1]
                prev_diag_diag = dp[row - 1, col - 1, 2]
                max_val = max(prev_right_diag, prev_down_diag, prev_diag_diag)
                if max_val != -np.inf:
                    dp[row, col, 2] = max_val + current_val

                    max_idx = np.argmax(
                        [prev_right_diag, prev_down_diag, prev_diag_diag]
                    )
                    backtrack[row, col, 2] = (row - 1, col - 1, max_idx)

    # Get the maximum score at the destination
    max_score = max(dp[num_rows - 1, num_cols - 1, :])

    # Determine which state gives the maximum score
    final_dir = np.argmax(dp[num_rows - 1, num_cols - 1, :])

    # Backtrack to find the path
    path = []
    cur_row, cur_col, cur_dir = num_rows - 1, num_cols - 1, final_dir
    while (cur_row, cur_col) != (0, 0) or not path:
        path.append((cur_row, cur_col))
        prev = backtrack[cur_row, cur_col, cur_dir]
        if prev is None:
            break
        cur_row, cur_col, cur_dir = prev
    path.append((0, 0))
    path.reverse()

    # import pdb; pdb.set_trace()  # fmt: skip
    return max_score, path


def _group_consecutive_alignments(
    alignments: list[tuple[int, int]],
) -> Alignment:
    """
    Group an alignment path of (i, j) index pairs into consecutive alignment blocks.

    Each output tuple contains two lists: the indices of source sentences `i`,
    and the indices of the aligned target sentences `j`.

    >>> _group_consecutive_alignments([(0, 0), (0, 1), (1, 2), (2, 3), (3, 3)])
    [([0], [0, 1]), ([1], [2]), ([2, 3], [3])]
    """

    grouped_alignments = []
    current_group = [alignments[0]]

    for prev_alignment, curr_alignment in zip(alignments, alignments[1:]):
        prev_src, prev_tgt = prev_alignment
        curr_src, curr_tgt = curr_alignment

        # If we're moving diagonally, start a new group
        if curr_src > prev_src and curr_tgt > prev_tgt:
            grouped_alignments.append(current_group)
            current_group = [curr_alignment]
        else:
            current_group.append(curr_alignment)

    grouped_alignments.append(current_group)

    # Convert grouped alignments to source-target mappings
    alignment_pairs = []
    for group in grouped_alignments:
        src_indices = sorted({i for i, j in group})
        tgt_indices = sorted({j for i, j in group})
        alignment_pairs.append((src_indices, tgt_indices))

    return alignment_pairs
