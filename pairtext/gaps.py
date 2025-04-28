import logging

import numpy as np

from pairtext.datatypes import AlignmentResult, Bound, Indices, Marker, SentenceList
from pairtext.utils import _concat_sentences, compute_similarity_matrix

logger = logging.getLogger(__name__)


def detect_gaps(
    initial_alignment: AlignmentResult,
    minscore_translation: float = 0.6,
    special_penalty: float = 0.025,
    concat_preference: float = 0.01,
) -> AlignmentResult:
    """
    Refine an initial sentence alignment by detecting and introducing gaps
    (1:0 or 0:1 alignments) using sentence similarity scores over two passes.

    Pass 2: Refine initial 1:n/n:1 blocks, add markers for Pass 3.
    Pass 3: Merge adjacent gaps and re-evaluate merged blocks.

    Args:
        initial_alignment: List of (source_indices, target_indices) tuples.
        minscore_translation: Minimum cosine similarity for an alignment.
        special_penalty: Pass 2 score threshold to override edge/concat preference.
        concat_preference: Pass 2 score margin to prefer full concatenation.

    Returns:
        A refined alignment list potentially including gaps.
    """
    # --- Pass 2 ---
    alignment_pass2 = _run_second_pass(
        initial_alignment=initial_alignment,
        minscore_translation=minscore_translation,
        special_penalty=special_penalty,
        concat_preference=concat_preference,
    )

    # --- Pass 3 Preparation ---
    merged_alignment, realignment_mask = _prepare_pass3_structure(alignment_pass2)

    # --- Pass 3 ---
    final_alignment = _run_third_pass(
        merged_alignment=merged_alignment,
        realignment_mask=realignment_mask,
        minscore_translation=minscore_translation,
    )

    logger.info("Gap detection finished.")
    return final_alignment


def _run_second_pass(
    initial_alignment: AlignmentResult,
    minscore_translation: float,
    special_penalty: float,
    concat_preference: float,
) -> AlignmentResult:
    """
    Perform the second pass of alignment refinement.
    Focus on initial 1:n and n:1 alignments, potentially splitting
    them into Gaps + Center + Gaps based on similarity, with special
    preference for edge/concatenated sentences. Add LEFT/RIGHT markers to gaps.

    Args:
        initial_alignment: The initial alignment from the first stage.
        minscore_translation: Minimum similarity score for a valid alignment.
        special_penalty: Score threshold to override edge/concat preference.
        concat_preference: Score margin to prefer full concatenation.

    Returns:
        Alignment result from pass 2, with markers added to gap segments.
    """
    logger.info("Performing Pass 2, refining 1:n and n:1 alignments...")
    pass_src_sentences: SentenceList = []
    pass_tgt_sentences: SentenceList = []
    candidate_map: list[tuple[list[Indices], list[Indices]] | None] = []

    # 1. Prepare candidates only for 1:n and n:1 pairs
    for src_indices, tgt_indices in initial_alignment:
        if len(src_indices) > 1 or len(tgt_indices) > 1:
            # Generate candidate sentences (single + concatenated)
            src_candidates, src_candidate_indices = _prepare_sentence_candidates(
                src_indices,
                initial_alignment.src_sentences,
            )
            tgt_candidates, tgt_candidate_indices = _prepare_sentence_candidates(
                tgt_indices,
                initial_alignment.tgt_sentences,
            )
            pass_src_sentences.extend(src_candidates)
            pass_tgt_sentences.extend(tgt_candidates)
            candidate_map.append((src_candidate_indices, tgt_candidate_indices))
        else:
            # Mark 1:1 pairs as not processed in this pass
            candidate_map.append(None)

    if not pass_src_sentences or not pass_tgt_sentences:
        logger.info("No 1:n or n:1 candidates found requiring gap detection.")
        return initial_alignment
        # We still need to check minscore_translation

    # 2. Calculate similarity matrix for all candidates
    similarity_matrix = compute_similarity_matrix(
        pass_src_sentences, pass_tgt_sentences, initial_alignment.model_name
    )

    # 3. Process each relevant alignment pair using the similarity matrix
    alignment_pass2: AlignmentResult = initial_alignment.copy(drop_result=True)

    src_offset = 0
    tgt_offset = 0
    for i, (original_src_indices, original_tgt_indices) in enumerate(initial_alignment):
        candidates = candidate_map[i]

        if candidates is None:
            # This pair wasn't processed (it was 1:1)
            assert len(original_src_indices) == len(original_tgt_indices) == 1
            score = initial_alignment.get_score(i)
            assert len(score) == 1
            alignment_pass2.add_pair(
                original_src_indices,
                original_tgt_indices,
                score_concat=score[0],
            )
            continue

        src_candidate_indices, tgt_candidate_indices = candidates
        num_src_candidates = len(src_candidate_indices)
        num_tgt_candidates = len(tgt_candidate_indices)

        # Extract the submatrix for this specific alignment pair
        src_slice = slice(src_offset, src_offset + num_src_candidates)
        tgt_slice = slice(tgt_offset, tgt_offset + num_tgt_candidates)
        submatrix = similarity_matrix[src_slice, tgt_slice]

        src_offset += num_src_candidates
        tgt_offset += num_tgt_candidates

        # Check if the best possible score meets the minimum threshold
        maxscore = np.max(submatrix)
        if maxscore < minscore_translation:  # Score too low -> 1:0 and 0:1
            # We'll deal with this in Pass 3
            alignment_pass2.add_pair(
                original_src_indices, original_tgt_indices, score_concat=maxscore
            )
            continue

        # --- Find the best matching pair ---
        n_src_singles = len(original_src_indices)
        n_tgt_singles = len(original_tgt_indices)

        first_idx = 0
        # Calculate indices (assuming n_src_singles >= 1 and n_tgt_singles >= 1)
        last_src_idx = n_src_singles - 1
        last_tgt_idx = n_tgt_singles - 1
        concat_src_idx = num_src_candidates - 1
        concat_tgt_idx = num_tgt_candidates - 1

        is_n_to_1 = n_src_singles > 1
        best_i, best_j = 0, 0  # Default indices

        if is_n_to_1:  # n:1 alignment
            first_score = submatrix[first_idx, 0]
            last_score = submatrix[last_src_idx, 0]
            concat_score = submatrix[concat_src_idx, 0]
            indices_map = [(first_idx, 0), (last_src_idx, 0), (concat_src_idx, 0)]
        else:  # 1:n alignment
            assert n_tgt_singles > 1  # consistency check
            first_score = submatrix[0, first_idx]
            last_score = submatrix[0, last_tgt_idx]
            concat_score = submatrix[0, concat_tgt_idx]
            indices_map = [(0, first_idx), (0, last_tgt_idx), (0, concat_tgt_idx)]

        scores_special = [first_score, last_score, concat_score]
        maxscore_special = max(scores_special)
        best_special_idx_in_list = np.argmax(scores_special)
        best_i, best_j = indices_map[best_special_idx_in_list]

        # Prefer concatenation slightly if scores are close
        if abs(concat_score - max(first_score, last_score)) < concat_preference:
            maxscore_special = concat_score
            best_i, best_j = indices_map[2]  # Index 2 = concat case

        # If the absolute best score is significantly better, override preference
        # maxscore = np.max(submatrix)  # already calculated earlier
        if (maxscore - maxscore_special) > special_penalty:
            best_i, best_j = np.unravel_index(submatrix.argmax(), submatrix.shape)
        # --- End of Pass 2 best match logic ---

        # Get the actual sentence indices corresponding to the best match
        best_src_indices = src_candidate_indices[best_i]
        best_tgt_indices = tgt_candidate_indices[best_j]

        # Split the original indices into gaps and the aligned center
        src_left, src_center, src_right = alignment_pass2.split_indices(
            original_src_indices, best_src_indices, with_marker=True
        )
        tgt_left, tgt_center, tgt_right = alignment_pass2.split_indices(
            original_tgt_indices, best_tgt_indices, with_marker=True
        )

        # Add the alignment pair to the final alignment
        if src_left:
            alignment_pass2.add_pair(src_left, [], score_concat=None)
        if tgt_left:
            alignment_pass2.add_pair([], tgt_left, score_concat=None)
        assert src_center and tgt_center  # Ensure we add the central alignment
        alignment_pass2.add_pair(src_center, tgt_center, score_concat=maxscore)
        # Note that we may be splitting based on maxscore_special,
        # but we need maxscore for minscore_translation in Pass 3.
        # This may be confusing for the user, possibly change later.
        if src_right:
            alignment_pass2.add_pair(src_right, [], score_concat=None)
        if tgt_right:
            alignment_pass2.add_pair([], tgt_right, score_concat=None)

        assert len(alignment_pass2) == len(alignment_pass2.scores_concat)
    return alignment_pass2


def _prepare_pass3_structure(
    alignment_pass2: AlignmentResult,
) -> tuple[AlignmentResult, list[str | None]]:
    """
    Process the alignment from pass 2, merging adjacent gap segments
    created with LEFT/RIGHT markers, and identify which segments need
    realignment in pass 3.
    Return the merged alignment and a mask indicating realignment need.
    """
    logger.info("Preparing structure for Pass 3, merging adjacent gaps...")
    merged_alignment: AlignmentResult = alignment_pass2.copy(drop_result=True)
    realignment_mask: list[str | None] = []

    if not alignment_pass2:
        return alignment_pass2, []

    i = 0
    while i < len(alignment_pass2):
        current_src, current_tgt = alignment_pass2[i]
        current_score = alignment_pass2.get_concat_score(i)
        merged_src, merged_tgt = list(current_src), list(current_tgt)  # copy
        realignment_type: str | None = None

        # Check marker, remove it if present
        current_src_marker = current_tgt_marker = None
        if current_src and isinstance(current_src[0], str):
            current_src_marker = merged_src.pop(0)
        if current_tgt and isinstance(current_tgt[0], str):
            current_tgt_marker = merged_tgt.pop(0)

        # Look ahead to the next segment for potential merges
        if i + 1 < len(alignment_pass2):
            next_src, next_tgt = alignment_pass2[i + 1]

            next_src_marker = next_tgt_marker = None
            if next_src and isinstance(next_src[0], str):
                next_src_marker = next_src[0]
            if next_tgt and isinstance(next_tgt[0], str):
                next_tgt_marker = next_tgt[0]

            # --- Merge Logic ---
            # Case 1: Right-Gap(SRC) followed by Left-Gap(SRC) -> Merge SRC
            if current_src_marker == Marker.R and next_src_marker == Marker.L:
                assert not current_tgt and not next_tgt
                merged_src.extend(next_src[1:])
                realignment_type = Bound.LR
                current_score = None
                i += 1  # Skip the next item

            # Case 2: Right-Gap(TGT) followed by Left-Gap(TGT) -> Merge TGT
            elif current_tgt_marker == Marker.R and next_tgt_marker == Marker.L:
                assert not current_src and not next_src
                merged_tgt.extend(next_tgt[1:])
                realignment_type = Bound.LR
                current_score = None
                i += 1  # Skip next

            # Case 3: Right-Gap(SRC) followed by Alignment(SRC, TGT) -> Prepend SRC to next alignment's SRC
            elif current_src_marker == Marker.R and next_src:
                assert not current_tgt
                assert next_tgt and next_src_marker is None
                # Move the gap into the next alignment segment for realignment;
                # modify the next item in the original list before it's processed
                alignment_pass2.update_pair(
                    i + 1, merged_src + next_src, next_tgt, score_concat=None
                )
                merged_src, merged_tgt = [], []
                # We'll mark next item for realignment in the next loop iteration

            # Case 4: Right-Gap(TGT) followed by Alignment(SRC, TGT) -> Prepend TGT to next alignment's TGT
            elif current_tgt_marker == Marker.R and next_tgt:
                assert not current_src
                assert next_src and next_tgt_marker is None
                alignment_pass2.update_pair(
                    i + 1, next_src, merged_tgt + next_tgt, score_concat=None
                )
                merged_src, merged_tgt = [], []

            # Case 5: Alignment(SRC, TGT) followed by Left-Gap(SRC) -> Append SRC to current alignment's SRC
            elif next_src_marker == Marker.L and current_src:
                assert current_tgt and current_src_marker is None
                assert not next_tgt
                merged_src.extend(next_src[1:])
                realignment_type = Bound.L
                current_score = None
                i += 1  # Skip next

            # Case 6: Alignment(SRC, TGT) followed by Left-Gap(TGT) -> Append TGT to current alignment's TGT
            elif next_tgt_marker == Marker.L and current_tgt:
                assert current_src and current_tgt_marker is None
                assert not next_src
                merged_tgt.extend(next_tgt[1:])
                realignment_type = Bound.L
                current_score = None
                i += 1  # Skip next

            # Case 7: Right-Gap(SRC) followed by Left-Gap(TGT) -> Merge into one SRC-gap, TGT-gap pair
            elif current_src_marker == Marker.R and next_tgt_marker == Marker.L:
                assert not current_tgt and not next_src
                merged_tgt.extend(next_tgt[1:])  # Add TGT gap indices
                realignment_type = Bound.LR
                current_score = None
                i += 1  # Skip next

            # Case 8: Right-Gap(TGT) followed by Left-Gap(SRC) -> Merge into one SRC-gap, TGT-gap pair
            elif current_tgt_marker == Marker.R and next_src_marker == Marker.L:
                assert not current_src and not next_tgt
                merged_src.extend(next_src[1:])  # Add SRC gap indices
                realignment_type = Bound.LR
                current_score = None
                i += 1  # Skip next

            # Default case (no merge): Handled by loop structure

            # Check need for realignment in cases 3/4 possibly followed by 5/6
            previous_src, previous_tgt = alignment_pass2[i - 1]
            if (previous_src and previous_src[0] == Marker.R) or (
                previous_tgt and previous_tgt[0] == Marker.R
            ):
                realignment_type = Bound.LR if realignment_type == Bound.L else Bound.R

        # Add the (potentially modified) segment
        if merged_src or merged_tgt:
            merged_alignment.add_pair(
                merged_src, merged_tgt, score_concat=current_score
            )
            realignment_mask.append(realignment_type)
        else:
            assert not realignment_type
        i += 1  # Move to next item

    # Sanity check
    merged_alignment.validate_indices()
    assert len(merged_alignment) == len(realignment_mask)
    return merged_alignment, realignment_mask


def _run_third_pass(
    merged_alignment: AlignmentResult,
    realignment_mask: list[str | None],
    minscore_translation: float,
) -> AlignmentResult:
    """
    Perform the third pass of alignment refinement.
    Re-evaluate similarity within blocks marked for realignment after Pass 2
    gap merging. Use simple best-score matching.

    Args:
        merged_alignment: Alignment after Pass 2 and gap merging.
        realignment_mask: List of required realignment type (LR, L, R).
        minscore_translation: Minimum similarity score for a valid alignment.

    Returns:
        The final refined alignment.
    """
    logger.info("Performing Pass 3, refining merged gap boundaries...")
    pass_src_sentences: SentenceList = []
    pass_tgt_sentences: SentenceList = []
    candidate_map: list[tuple[list[Indices], list[Indices]] | None] = []

    # 1. Prepare candidates only for pairs marked for realignment
    for i, (src_indices, tgt_indices) in enumerate(merged_alignment):
        if realignment_mask[i]:
            # Generate candidate sentences (single + concatenated)
            src_candidates, src_candidate_indices = _prepare_sentence_candidates(
                src_indices, merged_alignment.src_sentences, bound=realignment_mask[i]
            )
            tgt_candidates, tgt_candidate_indices = _prepare_sentence_candidates(
                tgt_indices, merged_alignment.tgt_sentences, bound=realignment_mask[i]
            )
            pass_src_sentences.extend(src_candidates)
            pass_tgt_sentences.extend(tgt_candidates)
            candidate_map.append((src_candidate_indices, tgt_candidate_indices))
        else:
            # Mark pairs not needing realignment as None
            candidate_map.append(None)

    if not pass_src_sentences or not pass_tgt_sentences:
        logger.info("No candidates requiring processing in Pass 3.")
        # return merged_alignment
        # We still need to check for minscore_translation
    else:
        # 2. Calculate similarity matrix for all candidates needing realignment
        similarity_matrix = compute_similarity_matrix(
            pass_src_sentences, pass_tgt_sentences, merged_alignment.model_name
        )

    # 3. Process each alignment pair
    final_alignment: AlignmentResult = merged_alignment.copy(drop_result=True)
    src_offset = 0
    tgt_offset = 0

    for i, (original_src_indices, original_tgt_indices) in enumerate(merged_alignment):
        candidates = candidate_map[i]

        if candidates is None:
            # This pair wasn't processed in this pass (no realignment needed)
            score = merged_alignment.get_concat_score(i)
            if score < minscore_translation:
                # Score too low, split into 1:0 and 0:1
                if original_src_indices:
                    final_alignment.add_pair(
                        original_src_indices, [], score_concat=None
                    )
                if original_tgt_indices:
                    final_alignment.add_pair(
                        [], original_tgt_indices, score_concat=None
                    )
            elif original_src_indices or original_tgt_indices:
                final_alignment.add_pair(
                    original_src_indices, original_tgt_indices, score_concat=score
                )
            continue

        # --- This pair needs processing ---
        src_candidate_indices, tgt_candidate_indices = candidates
        num_src_candidates = len(src_candidate_indices)
        num_tgt_candidates = len(tgt_candidate_indices)

        # Extract the submatrix
        src_slice = slice(src_offset, src_offset + num_src_candidates)
        tgt_slice = slice(tgt_offset, tgt_offset + num_tgt_candidates)
        submatrix = similarity_matrix[src_slice, tgt_slice]

        src_offset += num_src_candidates
        tgt_offset += num_tgt_candidates

        # Check minimum score threshold
        maxscore = np.max(submatrix) if submatrix.size > 0 else -1.0
        if maxscore < minscore_translation:
            # Score too low, split into 1:0 and 0:1
            if original_src_indices:
                final_alignment.add_pair(original_src_indices, [], score_concat=None)
            if original_tgt_indices:
                final_alignment.add_pair([], original_tgt_indices, score_concat=None)
            continue

        # Find the best matching pair using simple max score
        best_i, best_j = np.unravel_index(submatrix.argmax(), submatrix.shape)

        # Get the actual sentence indices corresponding to the best match
        best_src_indices = src_candidate_indices[best_i]
        best_tgt_indices = tgt_candidate_indices[best_j]

        # Split the original indices into gaps and the aligned center
        src_left, src_center, src_right = final_alignment.split_indices(
            original_src_indices, best_src_indices
        )
        tgt_left, tgt_center, tgt_right = final_alignment.split_indices(
            original_tgt_indices, best_tgt_indices
        )

        # Add results to final alignment
        if src_left:
            final_alignment.add_pair(list(src_left), [], score_concat=None)
        if tgt_left:
            final_alignment.add_pair([], list(tgt_left), score_concat=None)
        if src_center or tgt_center:  # Ensure we add the central alignment
            final_alignment.add_pair(
                list(src_center), list(tgt_center), score_concat=maxscore
            )
        if src_right:
            final_alignment.add_pair(list(src_right), [], score_concat=None)
        if tgt_right:
            final_alignment.add_pair([], list(tgt_right), score_concat=None)

    return final_alignment


def _prepare_sentence_candidates(
    indices: Indices,
    all_sentences: SentenceList,
    bound: str | None = None,
) -> tuple[SentenceList, list[Indices]]:
    """Concatenate sentences and their corresponding indices."""

    sents = [all_sentences[i] for i in indices]
    if len(sents) == 1:
        return sents, [indices]

    # `_concat_sentences` returns concatenated sentences and the groups
    # of *relative* indices used for concat. Example:
    # _concat_sentences(["a", "b", "c"]) -> (["a b", "b c", "a b c"], [[0, 1], [1, 2], [0, 1, 2]])
    concat_sents, relative_indices_concat = _concat_sentences(sents, bound=bound)

    all_candidate_sents = sents + concat_sents
    # Map relative indices back to original indices
    candidate_indices = [[i] for i in indices]  # Indices for single sentences
    for idx_list in relative_indices_concat:
        candidate_indices.append(
            [indices[i] for i in idx_list]
        )  # Indices for concatenated sentences

    return all_candidate_sents, candidate_indices
