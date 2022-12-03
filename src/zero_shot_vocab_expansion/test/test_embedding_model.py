from zero_shot_vocab_expansion import embedding_model


def test_embedding_model():
    model = embedding_model("bert-base-uncased")
    output = model.encode(["This is a test definition."])
    assert output.shape == (1, 768)
