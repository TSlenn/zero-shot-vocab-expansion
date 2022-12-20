from zero_shot_vocab_expansion.vocab_expansion import expand_vocabulary
from zero_shot_vocab_expansion.embedding_model import EmbeddingModel


def test_expand_vocabulary():
    new_words = ["valar", "maiar", "qasar", "albatross"]
    definitions = {
        "maiar": ["The Maiar (singular Maia) were primordial spirits created to help the Valar first shape the World"] # noqa
    }
    emb_model = EmbeddingModel(backbone_model="bert-base-uncased")
    mod, tok = expand_vocabulary(
        new_vocab=new_words,
        definitions=definitions,
        model="bert-base-uncased",
        embedding_model=emb_model
    )
    assert len(tok) == 30526
