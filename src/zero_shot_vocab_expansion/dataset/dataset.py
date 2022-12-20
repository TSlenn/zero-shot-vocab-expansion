import json
import numpy as np
from pathlib import Path
import random
from torch.utils.data import Dataset
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer, InputExample
from typing import Union
from .utils import get_definitions
from ..evaluation import MSEDefinitionEvaluator


def _update_definitions(words: list, definitions: dict):
    """Update fn for VocabDataset."""
    definitions = [
        definitions[word] if word in definitions else get_definitions(word)
        for word in words
    ]
    # Remove words with no definitions.
    definitions = {
        word: defs for word, defs in zip(words, definitions)
        if len(defs) > 0
    }
    words = list(definitions.keys())
    return words, definitions


class VocabDataset(Dataset):
    """Dataset of word definitions and corresponding embeddings.

    VocabDataset extracts vocabulary from a transformers tokenizer or
    SentenceTransformer, and matches them with a target embedding from
    a transformers model or SentenceTransformer backbone model. Word
    definitions are discovered using nltk wordnet synsets, or user provided
    definitions.

    VocabDataset can be initilized with:
        - VocabDataset(config_dict)
        - VocabDataset.from_config(config_dict)
        - VocabDataset.from_model(model_name or model object)
        - VocabDataset.load("path/to/config.json")
        - VocabDataset.from_list(list_of_words)

    Attributes:
        select_definition (str): Selection method used in __getitem__.
            Valid options: ["random", "best"].
        embeddings (Tensor): Token embeddings.
        token_map (dict): Dictionary of {word: int} that maps words to the
            correct embedding index.
        definitions (dict): Dictionary of {word: List[definitions]}
        words (list): Words used to construct the dataset.

    """
    def __init__(self, config: dict):
        """Initializes dataset from config dict."""
        if config["embeddings"] is not None:
            embeddings = Tensor(config["embeddings"])
        else:
            embeddings = None
        self.select_definition = config["select_definition"]
        self.embeddings = embeddings
        self.token_map = config["token_map"]
        self.definitions = config["definitions"]
        self.words = config["words"]

    @classmethod
    def from_config(cls, config: dict):
        """Initializes from config dictionary."""
        # Default constructor, included alias so .from_config works.
        return cls(config)

    @classmethod
    def from_model(
        cls,
        model: Union[str, PreTrainedModel, SentenceTransformer],
        tokenizer: PreTrainedTokenizerBase = None,
        definitions: dict = {},
        select_definition: str = "best",
    ):
        """Finds model vocab and corresponding word definitions.

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

        """
        msg = "select_definition must be one of ['best', 'random']"
        assert select_definition in ["best", "random"], msg
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
        words = list(tokenizer.vocab.keys())
        words, definitions = _update_definitions(words, definitions)
        config = {
            "select_definition": select_definition,
            "embeddings": mod.embeddings.word_embeddings.weight.detach(),
            "token_map": tokenizer.vocab,
            "words": words,
            "definitions": definitions
        }

        return cls(config)

    @classmethod
    def load(cls, filepath: str):
        """Loads dataset object attributes from save file."""
        filepath = Path(filepath).with_suffix(".json")
        with open(filepath, "r") as f:
            load_dict = json.load(f)
        return cls(load_dict)

    @classmethod
    def from_list(
        cls,
        words: list,
        definitions: dict,
        select_definition: str = "best",
    ):
        """Initializes unsupervised dataset from a list of words.

        Args:
            words (list of str): List of words that will be paired with
                definitions.
            definitions (dict): Dictionary of user provided word definitions.
                These are prioritized over wordnet definitions.
                {"word": ["definition1", "definition2"]}
            select_definition (str): "best" or "random. If "best", the top
                definition will be used. If "random", definition will be
                randomly selected from available definitions.

        """
        words, definitions = _update_definitions(words, definitions)
        config = {
            "select_definition": select_definition,
            "embeddings": None,
            "token_map": None,
            "words": words,
            "definitions": definitions
        }
        return cls(config)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        definitions = self.definitions[word]
        if self.select_definition == "best":
            definition = definitions[0]
        elif self.select_definition == "random":
            definition = random.choice(definitions)
        if self.embeddings is not None:
            embedding = self.embeddings[self.token_map[word]].tolist()
        else:
            embedding = None
        return InputExample(guid=word, texts=[definition], label=embedding)

    def get_config(self):
        return {
            "select_definition": self.select_definition,
            "embeddings": self.embeddings.tolist().copy(),
            "token_map": self.token_map.copy(),
            "definitions": self.definitions.copy(),
            "words": self.words.copy()
        }

    def save(self, filepath):
        """Saves dataset attributes to directory."""
        filepath = Path(filepath).with_suffix(".json")
        save_dict = self.get_config()
        with open(filepath, "w") as f:
            json.dump(save_dict, f)

    def to_evaluator(self, name="", write_csv: bool = False):
        definitions = list()
        embeddings = list()
        for word in self.words:
            definitions.append(self.definitions[word])
            embeddings.append(self.embeddings[self.token_map[word]].numpy())
        embeddings = np.stack(embeddings)
        evaluator = MSEDefinitionEvaluator(
            definitions, embeddings, name, write_csv
        )
        return evaluator


def split_dataset(ds: VocabDataset, split: float, shuffle: bool = True):
    """Splits VocabDataset into two parts for train and val.

    Args:
        ds (VocabDataset): VocabDataset object.
        split (float): Between 0 and 1. Split ratio between the primary
            and secondary dataset.

    """
    config = ds.get_config()
    ds1 = VocabDataset.from_config(config)
    ds2 = VocabDataset.from_config(config)

    # split up words
    words = config["words"]
    if shuffle:
        random.shuffle(words)
    n_1 = int(len(words)*split)
    ds1.words = words[:n_1]
    ds2.words = words[n_1:]

    return ds1, ds2
