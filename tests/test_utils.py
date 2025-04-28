import numpy as np

from pairtext.utils import compute_similarity_matrix

# Language: python


class FakeTensor:
    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array


def fake_similarity(src_embeddings, tgt_embeddings):
    # Compute dot product for cosine similarity as our fake similarity
    return FakeTensor(np.dot(src_embeddings, tgt_embeddings.T))


def fake_encode(sentences):
    # Return different embeddings based on number of sentences
    if len(sentences) == 2:
        return np.array([[1, 0], [0, 1]])
    elif len(sentences) == 1:
        return np.array([[1, 0]])
    else:
        # For any other case, simply return a default array based on input length
        return np.array([np.ones(2) for _ in sentences])


def test_compute_similarity_matrix(mocker):
    # Create fake model using pytest's mocker
    fake_model = mocker.Mock()
    fake_model.encode.side_effect = fake_encode
    fake_model.similarity.side_effect = fake_similarity

    # Test with list input
    source = ["Hello", "World"]
    target = ["Hola"]

    # Expected: np.dot([[1,0],[0,1]], [[1,0]].T) = [[1],[0]]
    expected = np.array([[1], [0]])
    result = compute_similarity_matrix(source, target, fake_model)
    assert np.array_equal(result, expected)

    # Test with single string input (should be converted to list)
    source_str = "Hello"
    target_str = "Hola"
    # fake_encode for both returns 1D array wrapped to 2D, expecting dot product [[1]]
    expected_single = np.array([[1]])
    result_single = compute_similarity_matrix(source_str, target_str, fake_model)
    assert np.array_equal(result_single, expected_single)
