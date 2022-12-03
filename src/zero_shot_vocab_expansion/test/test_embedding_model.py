from zero_shot_vocab_expansion import EmbeddingModel


def test_EmbeddingModel():
    model = EmbeddingModel("bert-base-uncased")
    output = model.encode(["This is a test definition."])
    assert output.shape == (1, 768)


def test_frozen_EmbeddingModel():
    model = EmbeddingModel("bert-base-uncased", freeze_backbone=True)
    output = model.encode(["This is a test definition."])
    assert output.shape == (1, 768)
