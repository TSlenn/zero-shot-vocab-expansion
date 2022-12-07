from zero_shot_vocab_expansion.dataset import VocabDataset, split_dataset
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from math import isclose


def test_VocabDataset_transformers():
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = VocabDataset(model, tokenizer)
    assert len(ds) > 0
    output = ds[0]
    assert isinstance(output.guid, str)
    assert isinstance(output.texts[0], str)
    assert isinstance(output.label, list)
    # confirm method works
    _ = ds.to_evaluator()


def test_VocabDataset_str():
    ds = VocabDataset("bert-base-uncased")
    assert len(ds) > 0
    output = ds[0]
    assert isinstance(output.guid, str)
    assert isinstance(output.texts[0], str)
    assert isinstance(output.label, list)


def test_VocabDataset_ST():
    model = SentenceTransformer("bert-base-uncased")
    ds = VocabDataset(model)
    assert len(ds) > 0
    output = ds[0]
    assert isinstance(output.guid, str)
    assert isinstance(output.texts[0], str)
    assert isinstance(output.label, list)


def test_VocabDataset_custom_defs():
    defs = {"test": ["hello world"]}
    ds = VocabDataset("bert-base-uncased", definitions=defs)
    assert ds.definitions["test"][0] == "hello world"


def test_split_dataset():
    ds = VocabDataset("bert-base-uncased")
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
