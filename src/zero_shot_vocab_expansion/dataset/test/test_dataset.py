from zero_shot_vocab_expansion.dataset import VocabDataset
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


def test_VocabDataset_transformers():
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = VocabDataset(model, tokenizer)
    assert len(ds) > 0
    output = ds[0]
    assert "word" in output
    assert "definition" in output
    assert "target" in output


def test_VocabDataset_str():
    ds = VocabDataset("bert-base-uncased")
    assert len(ds) > 0
    output = ds[0]
    assert "word" in output
    assert "definition" in output
    assert "target" in output


def test_VocabDataset_ST():
    model = SentenceTransformer("bert-base-uncased")
    ds = VocabDataset(model)
    assert len(ds) > 0
    output = ds[0]
    assert "word" in output
    assert "definition" in output
    assert "target" in output


def test_VocabDataset_custom_defs():
    defs = {"test": ["hello world"]}
    ds = VocabDataset("bert-base-uncased", definitions=defs)
    assert ds.definitions["test"][0] == "hello world"
