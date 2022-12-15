from zero_shot_vocab_expansion.vocab_expansion.initializers import (
    get_distribution_params,
    random_initializer
)
import numpy as np
from transformers import AutoModel

# Get embeddings from an existing model
MODEL_NAME = "bert-base-uncased"
MODEL = AutoModel.from_pretrained(MODEL_NAME)
EMBEDDINGS = MODEL.embeddings.word_embeddings.weight.detach()


def test_get_distribution_params():
    """Asserts function returns correct mean, covariance."""
    X = np.array([[2, 4, 6, 8, 10],
                  [7, 3, 5, 1, 9]]).T
    mu, cov = get_distribution_params(X)
    assert np.array_equal(mu, np.array([6., 5.]))
    assert np.array_equal(cov, np.array([[8., 0.8], [0.8, 8.]]))


def test_random_initializer():
    """Checks that function returns (n, k) numpy array."""
    new_embeddings = random_initializer(EMBEDDINGS.numpy(), 10)
    assert new_embeddings.shape[0] == 10
    assert new_embeddings.shape[1] == EMBEDDINGS.shape[1]
