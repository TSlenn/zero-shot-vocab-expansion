from .initializers import random_initializer, model_initializer
from ..embedding_model import EmbeddingModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np
import torch


def expand_vocabulary(
    new_vocab: list,
    model: Union[str, PreTrainedModel, SentenceTransformer],
    tokenizer: PreTrainedTokenizerBase = None,
    method: str = "model",
    embedding_model: EmbeddingModel = None,
    definitions: dict = {},
):
    """Adds new vocabulary to tokenizer and transformer.

    Args:
        new_vocab (list of str): New words to add to the tokenizer and model.
        model (str, PreTrainedModel, SentenceTransformer): Pretrained model,
            model name, or path to a local model.
        tokenizer (PreTrainedTokenizerBase): Optional. Tokenizer for model.
            If model is provided as str or SentenceTransformer, the
            corresponding tokenizer will be loaded automatically.
        method (str): Method to initialize new embeddings. Valid options are
            "model" or "random". Defaults to "model".
        embedding_model (EmbeddingModel): Model to use to initialize new
            embeddings. Only required if method is "model".
        definitions (dict):  Dictionary of user provided word definitions.
            These are prioritized over wordnet definitions.

    Returns:
        PreTrainedModel, PreTrainedTokenizerBase

    """
    # handle various options for model input
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
    old_embeddings = mod.embeddings.word_embeddings.weight.detach()

    # Make sure nothing in new_vocab is already in the tokenizer
    old_vocab = set(tokenizer.get_vocab())
    new_vocab = set(new_vocab).difference(old_vocab)
    new_embeddings = dict()

    if method == "model":
        assert embedding_model is not None, (
            "embedding_model must defined for model initialization"
        )
        new_embeddings.update(
            model_initializer(embedding_model, list(new_vocab), definitions)
        )
        # update new_vocab to only include not found by model
        new_vocab = new_vocab.difference(set(new_embeddings))

    # initialize remaining words with random initialization
    if len(new_vocab) > 0:
        rand_embeddings = random_initializer(
            old_embeddings.numpy(), len(new_vocab)
        )
        new_embeddings.update(
            {word: emb for word, emb in zip(new_vocab, rand_embeddings)}
        )

    # Add the new words and embeddings to model and tokenizer
    embeddings = np.stack(list(new_embeddings.values()), axis=0)
    words = list(new_embeddings.keys())
    tokenizer.add_tokens(words)

    mod.resize_token_embeddings(len(tokenizer))
    state_dict = mod.state_dict()
    emb_layer_name = [
        x for x in state_dict if "embeddings.word_embeddings.weight" in x
    ][0]
    # overwrite new embeds with our initializations
    state_dict[emb_layer_name][-len(words):, :] = torch.Tensor(embeddings)
    mod.load_state_dict(state_dict)

    return mod, tokenizer
