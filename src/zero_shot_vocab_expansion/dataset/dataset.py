import json
from pathlib import Path
import random
from torch.utils.data import Dataset
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import InputExample
from typing import Union
from .utils import get_definitions


class VocabDataset(Dataset):
    """Dataset of word definitions and corresponding embeddings."""
    def __init__(
        self,
        model: Union[str, PreTrainedModel, SentenceTransformer] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        definitions: dict = {},
        select_definition: str = "best",
        load_path: str = None
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
            select_definition (str): "best" or "random. If "best", the top
                definition will be used. If "random", definition will be
                randomly selected from available definitions.
            load_path (str): If provided, all other args will be ignored and
                the dataset object will be loaded from the provided path.

        """
        if load_path is not None:
            self.load(load_path)
            return
        if model == "rosebud":
            # hacky secret code to skip the init steps
            return
        msg = "select_definition must be one of ['best', 'random']"
        assert select_definition in ["best", "random"], msg
        self.select_definition = select_definition
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
        if self.select_definition == "best":
            definition = definitions[0]
        elif self.select_definition == "random":
            definition = random.choice(definitions)
        embedding = self.embeddings[self.token_map[word]]
        return InputExample(guid=word, texts=[definition], label=embedding)

    def save(self, filepath):
        """Saves dataset attributes to directory."""
        filepath = Path(filepath).with_suffix(".json")
        save_dict = {
            "select_definition": self.select_definition,
            "embeddings": self.embeddings.tolist(),
            "token_map": self.token_map,
            "definitions": self.definitions,
            "words": self.words
        }
        with open(filepath, "w") as f:
            json.dump(save_dict, f)

    def load(self, filepath):
        """Loads dataset object attributes from save file."""
        filepath = Path(filepath).with_suffix(".json")
        with open(filepath, "r") as f:
            load_dict = json.load(f)
        self.select_definition = load_dict["select_definition"]
        self.embeddings = Tensor(load_dict["embeddings"])
        self.token_map = load_dict["token_map"]
        self.definitions = load_dict["definitions"]
        self.words = load_dict["words"]


def split_dataset(ds: VocabDataset, split: float, shuffle: bool = True):
    """Splits VocabDataset into two parts for train and val.

    Args:
        ds (VocabDataset): VocabDataset object.
        split (float): Between 0 and 1. Split ratio between the primary
            and secondary dataset.

    """
    ds1 = VocabDataset("rosebud")
    ds2 = VocabDataset("rosebud")

    # split up words
    words = ds.words.copy()
    if shuffle:
        random.shuffle(words)
    n_1 = int(len(words)*split)
    ds1.words = words[:n_1]
    ds2.words = words[n_1:]

    # copy all other attributes
    ds1.select_definition = ds.select_definition
    ds1.embeddings = ds.embeddings
    ds1.token_map = ds.token_map
    ds1.definitions = ds.definitions
    ds2.select_definition = ds.select_definition
    ds2.embeddings = ds.embeddings
    ds2.token_map = ds.token_map
    ds2.definitions = ds.definitions

    return ds1, ds2
