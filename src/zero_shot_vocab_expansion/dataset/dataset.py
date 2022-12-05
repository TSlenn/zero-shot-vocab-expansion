from torch.utils.data import Dataset
import random
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer
from typing import Union
from .utils import get_definitions


class VocabDataset(Dataset):
    """Dataset of word definitions and corresponding embeddings."""
    def __init__(
        self,
        model: Union[str, PreTrainedModel, SentenceTransformer],
        tokenizer: PreTrainedTokenizerBase = None,
        definitions: dict = {}
    ):
        """Finds model vocab and corresponding word definitions.

        VocabDataset extracts vocabulary from a transformers tokenizer or
        SentenceTransformer, and matches them with a target embedding from
        a transformers model or SentenceTransformer backbone model. Word
        definitions are discovered using nltk wordnet synsets, or user provided
        definitions.

        Args:
            model (str, PreTrainedModel, SentenceTransformer): Pretrained
                model, model name, or path to a local model.
            tokenizer (PreTrainedTokenizerBase): Optional. Tokenizer for model.
                If model is provided as str or SentenceTransformer, the
                corresponding tokenizer will be loaded automatically.
            definitions (dict): Dictionary of user provided word definitions.
                These are prioritized over wordnet definitions.
                {"word": ["definition1", "definition2"]}
        """
        # get transformers model, tokenizer
        if isinstance(model, str):
            mod = AutoModel.from_pretrained(model)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model)
        elif isinstance(model, PreTrainedModel):
            mod = model
            msg = "tokenizer must be provided with transformer model"
            assert tokenizer is not None, msg
        elif isinstance(model, SentenceTransformer):
            tokenizer = model.tokenizer
            mod = model._first_module().auto_model

        # get word definitions and target embeddings
        self.embeddings = mod.embeddings.word_embeddings.weight.detach()
        self.token_map = tokenizer.vocab
        words = list(self.token_map.keys())
        definitions = [
            definitions[word] if word in definitions else get_definitions(word)
            for word in words
        ]
        self.definitions = {
            word: defs for word, defs in zip(words, definitions)
            if len(defs) > 0
        }
        self.words = list(self.definitions.keys())

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        definitions = self.definitions[word]
        embedding = self.embeddings[self.token_map[word]]
        return {
            "word": word,
            "definition": random.choice(definitions),
            "target": embedding
        }
