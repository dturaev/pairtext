import numpy as np
import pytest

from pairtext.datatypes import Alignment, AlignmentResult, SentenceList


@pytest.fixture
def alignment_result():
    # Alignment is a list of tuples: ([src_indices], [tgt_indices])
    alignment: Alignment = [([0], [0]), ([1], [1])]
    src_sentences: SentenceList = ["The cat", "has stripes"]
    tgt_sentences: SentenceList = ["Die Katze", "ist gestreift"]
    matrix = np.array([[0.94103366, 0.1973314], [0.23843953, 0.76683253]])
    model_name = "test-model"
    scores = [[0.94], [0.77]]
    scores_concat = [0.9, 0.8]  # Example concatenated scores

    return AlignmentResult(
        _pairs=alignment,
        src_sentences=src_sentences,
        tgt_sentences=tgt_sentences,
        matrix=matrix,
        model_name=model_name,
        scores=scores,
        scores_concat=scores_concat,
    )


def test_len(alignment_result):
    assert len(alignment_result) == 2


def test_getitem(alignment_result):
    assert alignment_result[0] == ([0], [0])
    assert alignment_result[1] == ([1], [1])


def test_iter(alignment_result):
    pairs = [pair for pair in alignment_result]
    assert pairs == alignment_result.result


def test_result_property(alignment_result):
    assert alignment_result.result == alignment_result._pairs


def test_get_score(alignment_result):
    assert alignment_result.get_score(0) == [0.94]
    assert alignment_result.get_score(1) == [0.77]
    with pytest.raises(IndexError):
        _ = alignment_result.get_score(2)


def test_get_pair_sentences(alignment_result):
    src, tgt = alignment_result.get_pair_sentences(0)
    assert src == ["The cat"]
    assert tgt == ["Die Katze"]


def test_validate_indices_valid(alignment_result):
    # Given the valid indices fixture, validate_indices should return True
    assert alignment_result.validate_indices() is True


def test_validate_indices_invalid(alignment_result):
    # Create an AlignmentResult with an invalid source index (2 is out of range)
    invalid_alignment: Alignment = [([0], [0]), ([2], [1])]
    with pytest.raises(ValueError):
        # This raises a ValueError in __post_init__ since _pairs and scores lengths differ
        # OR you can create the instance without scores to test validate_indices directly.
        AlignmentResult(
            _pairs=invalid_alignment,
            src_sentences=alignment_result.src_sentences,
            tgt_sentences=alignment_result.tgt_sentences,
            matrix=alignment_result.matrix,
            model_name=alignment_result.model_name,
            scores=[[0.9], [0.8]],
        )


def test_copy_full(alignment_result):
    # copy with drop_result=False
    ar_copy = alignment_result.copy(drop_result=False)
    # check that both have the same content
    assert ar_copy._pairs == alignment_result._pairs
    assert ar_copy.scores == alignment_result.scores
    assert ar_copy.scores_concat == alignment_result.scores_concat

    # modify the copy and ensure the original is not affected
    ar_copy._pairs.append(([1, 2], [1]))
    assert len(alignment_result._pairs) == 2


def test_copy_drop_result(alignment_result):
    # copy with drop_result=True
    ar_copy = alignment_result.copy(drop_result=True)
    assert ar_copy._pairs == []
    assert ar_copy.scores is None
    assert ar_copy.scores_concat is None


def test_update_pair_success(alignment_result):
    # update an alignment pair with new indices and scores
    alignment_result.update_pair(0, [1], [2], score=10.1, score_concat=11.1)
    assert alignment_result._pairs[0] == ([1], [2])
    assert alignment_result.scores[0] == [10.1]
    assert alignment_result.scores_concat[0] == 11.1


def test_update_pair_without_scores(alignment_result):
    # remove scores and score_concat to force error
    alignment_result.scores = None
    with pytest.raises(ValueError):
        alignment_result.update_pair(0, [1], [2], score=0.9)

    # reset scores for next test and remove scores_concat
    alignment_result.scores = [[0.5], [0.6]]
    alignment_result.scores_concat = None
    with pytest.raises(ValueError):
        alignment_result.update_pair(0, [1], [2], score_concat=1.1)
