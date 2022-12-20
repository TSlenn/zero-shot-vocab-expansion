from zero_shot_vocab_expansion.vocab_expansion.initializers import (
    get_distribution_params,
    random_initializer,
    model_initializer
)
import numpy as np
from zero_shot_vocab_expansion.embedding_model import EmbeddingModel


# Get embeddings from an existing model
MODEL_NAME = "bert-base-uncased"
MODEL = EmbeddingModel(MODEL_NAME)
EMBEDDINGS = MODEL._first_module().auto_model.embeddings.word_embeddings.weight.detach() # noqa


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


def test_model_initializer():
    words = ["barbecue", "chips", "penguin", "marsupial",
             "kerbonaut", "blackberry", "askldfjadjl"]
    definitions = {
        "kerbonaut": ["a cartoon astronaut from the game Kerbal Space Program"]
    }
    embeddings = model_initializer(MODEL, words, definitions)
    assert len(embeddings) == 6
    assert embeddings["barbecue"].shape[0] == 768
