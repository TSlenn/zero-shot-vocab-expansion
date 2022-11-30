from zero_shot_vocab_expansion.dataset.utils import get_definitions


def test_get_definitions():
    defs = get_definitions("strawberry")
    assert len(defs) > 0
