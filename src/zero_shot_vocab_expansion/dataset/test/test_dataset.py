from zero_shot_vocab_expansion.dataset import VocabDataset, split_dataset
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from math import isclose


def test_VocabDataset_transformers():
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = VocabDataset.from_model(model, tokenizer)
    assert len(ds) > 0
    output = ds[0]
    assert isinstance(output.guid, str)
    assert isinstance(output.texts[0], str)
    assert isinstance(output.label, list)
    # confirm method works
    _ = ds.to_evaluator()

    # Check that we can initialize with a config
    ds2 = VocabDataset.from_config(ds.get_config())
    ds2.words = ds2.words[:10]
    assert len(ds2) == 10
    # Ensure original dataset was not altered
    assert len(ds) > 10
    output = ds2[0]
    assert isinstance(output.guid, str)
    assert isinstance(output.texts[0], str)
    assert isinstance(output.label, list)


def test_VocabDataset_str():
    ds = VocabDataset.from_model("bert-base-uncased")
    assert len(ds) > 0
    output = ds[0]
    assert isinstance(output.guid, str)
    assert isinstance(output.texts[0], str)
    assert isinstance(output.label, list)


def test_VocabDataset_ST():
    model = SentenceTransformer("bert-base-uncased")
    ds = VocabDataset.from_model(model)
    assert len(ds) > 0
    output = ds[0]
    assert isinstance(output.guid, str)
    assert isinstance(output.texts[0], str)
    assert isinstance(output.label, list)


def test_VocabDataset_custom_defs():
    defs = {"test": ["hello world"]}
    ds = VocabDataset.from_model("bert-base-uncased", definitions=defs)
    assert ds.definitions["test"][0] == "hello world"


def test_VocabDataset_from_list():
    words = ["apple", "banana", "cranberry"]
    definitions = {
        "apple": ["test definition"]
    }
    ds = VocabDataset.from_list(words, definitions)
    assert len(ds) == 3
    output = ds[0]
    assert isinstance(output.guid, str)
    assert isinstance(output.texts[0], str)


def test_split_dataset():
    ds = VocabDataset.from_model("bert-base-uncased")
    x, y = split_dataset(ds, 0.5)
    assert isclose(len(x) / len(y), 1, abs_tol=0.05)
    # confirm that the individual datasets work
    outputx = x[0]
    outputy = y[0]
    assert isinstance(outputx.guid, str)
    assert isinstance(outputx.texts[0], str)
    assert isinstance(outputx.label, list)
    assert isinstance(outputy.guid, str)
    assert isinstance(outputy.texts[0], str)
    assert isinstance(outputy.label, list)
