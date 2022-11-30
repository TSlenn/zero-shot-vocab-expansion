from zero_shot_vocab_expansion.dataset import VocabDataset
from transformers import AutoModel, AutoTokenizer

MODEL = AutoModel.from_pretrained("bert-base-uncased")
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")


def test_VocabDataset():
    ds = VocabDataset(MODEL, TOKENIZER)
    assert len(ds) > 0
    output = ds[0]
    assert "word" in output
    assert "definition" in output
    assert "target" in output
