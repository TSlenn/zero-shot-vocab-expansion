## Initial Setup

setup.cfg contains all of the required pip packages and will automatically install them
when you pip install this repository. If you intend to use GPU
for training, you should manually install pytorch first to ensure you get the
version that is compatible with your CUDA version. This repo alse uses "wordnet"
to look up word definitions, which requires some nltk downloads after the pip
installations are completed. The below example uses CUDA 11.6.

```
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install .
python -c "import nltk;nltk.download('wordnet')"
python -c "import nltk;nltk.download('omw-1.4')"
```

## Embedding Model

The model that will be used to generate new token embeddings will have:
- Transformer backbone
- Mean Pooling to get a sentence embedding
- Dense projection heads to predict new embedding

The model can be initialized with the name of a Huggingface model, or a path
to a locally saved model.

```
from zero_shot_vocab_expansion import EmbeddingModel


model = EmbeddingModel(backbone_model = "bert-base-uncased")
```

EmbeddingModel has 3 additional optional args:
- max_seq_length: Maximum token length for the model. Defaults to 512.
- dense_layers: Number of dense layers in the projection head.
- freeze_backbone: If True, freezes the weights in the transformer model.
    Defaults to False.

freeze_backbone is useful if you have limited data or limited compute.  
If you'd like to freeze the backbone after initializing the model:

```
model._freeze_backbone()
```

## Dataset

To build a supervised dataset to train your Emedding Model, use the VocabDataset class.
VocabDataset builds a dataset of words and associated word definitions, as well as the
target embeddings for the word. The objective of the Dataset/Model combo is to create a
model that predict/initialize the embeddings of new vocabulary based on the new word
definitions.

```
from zero_shot_vocab_expansion.dataset import VocabDataset
```

VocabDataset is built as a sentence-transformers dataset. The output of the __getitem__
method is a InputExample with attributes "guid", "texts", and "label". For VocabDataset,
"guid" is the word, "texts" is the word definition, and "label" is the target embedding
vector.

Attributes:
    select_definition (str): Selection method used in __getitem__.
        Valid options: ["random", "best"].
    embeddings (Tensor): Target embeddings.
    token_map (dict): Dictionary of {word: int} that maps vocabulary to the
        correct embedding index.
    definitions (dict): Dictionary of {word: List[definitions]}
    words (list): Vocabulary used to construct the dataset.

There are multiple ways to construct your VocabDataset:

### Using a pretrained model / tokenizer

This constructor makes it very easy to build a embedding dataset from your pre-existing
model. You can simply feed it the name of Huggingface Model, a transformers model object
and tokenizer, or a SentenceTransformers model.

```
# Initialize using a model name (or path to model)
ds = VocabDataset.from_model("bert-base-uncased")

# or using a transformers model
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ds = VocabDataset.from_model(model=model, tokenizer=tokenizer)

# or using a sentence-transformers model
from sentence_transformers import SentenceTransformer
ds = SentenceTransformer("bert-base-uncased")
```

Additional kwargs:
    definitions: {word: ["definition1", "definition2"]} dictionary of user provided definitions.
    select_definition:  Selection method used in __getitem__. Valid options: ["random", "best"].

Using this constructor method, VocabDataset will:
- extract vocabulary list from the model's tokenizer
- extract target embeddings from the model's embedding layer
- finds definitions for extracted vocabulary using nltk.wordnet synsets
- assigns user defined word definitions via the optional "definitions" kwarg
- removes vocabulary with no known definitions
- save the embeddings, token_map, definitions, and words as object attributes

### From a configuration dictionary

This constructor allows you initialize a VocabDataset from a configuration dictionary.
This method allows you to reload a previously built VocabDataset, or to fully customize
your own dataset. The config dict must contain the 5 attributes used by the dataset.

config_dict = {
    "select_definitions": <"random" or "best">,
    "embeddings": <Tensor or nested list>,  # of embedding vectors>,
    "token_map": <dictionary>,              # {str: int} dictionary that maps vocabulary to embedding row
    "definitions": <dictionary>,            # definitions for each word
    "words": <list>                         # list of words that have keys in definitions
}

```
ds = VocabDataset(config_dict)
# or
ds = VocabDataset.from_config(config_dict)
```
The default __init__ constructor builds from a config dict. The .from_config() constructor
method was included to support explicit, readable code.

### From a saved configuration json file

This is identical to the .from_config constructor, except it reads from a saved
json file.

```
ds = VocabDataset.load("config.json")
```

### From a list of words

This constructor is intended for inference on new vocabulary. Instead of using the vocabulary
from a pretrained tokenizer, you can feed it a list of new words from your corpus that you
want to add to a model, along with any definitions that you want to manually define. It will
search nltk.wordnet for definitions to the new vocabulary, and add anything that it finds
to the dataset.

WARNING: This method does not initialize target embeddings. A dataset build using this method
will not be usable for training.

```
words = ["apple", "banana", "cranberry"]
definitions = {
    "apple": ["test definition"]
}
ds = VocabDataset.from_list(words, definitions)
```

### Methods

get_config(): returns configuration dictionary for the dataset.  
save("path/to/config.json"): saves configuration dictionary to json file.  
to_evaluator(): Creates a sentence-transformers evaluator object that can be used during training.


### Supporting Utility Functions

The split_dataset() function lets you randomly split a VocabDataset into a train/val set.

```
from zero_shot_vocab_expansion.dataset import split_dataset


train_ds, val_ds = split_dataset(ds, 0.5)
```


## Training your model

This repo uses the sentence-transformers library to build and train embedding models.
You can find an example of how to train a model in sample-notebooks/training.ipynb

## Adding new vocabulary to a pretrained model

Once you have a trained EmbeddingModel, you can use it to add new vocabulary to your
transformer.

```
# Two fictional words, and two real words that can be found in wordnet, but are not in BERT
new_words = ["valar", "maiar", "qasar", "albatross"]
definitions = {
    "valar": ["The Valar (Quenya; singular Vala) were the Powers of Arda who governed the world under the direction of Ilúvatar. They dwelt on the western continent of Aman."],
    "maiar": ["The Maiar (singular Maia) were primordial spirits created to help the Valar first shape the World"]
}
emb_model = <your trained EmbeddingModel>

# Use the embedding model to add your new vocab to bert-base-uncased
mod, tok = expand_vocabulary(
    new_vocab=new_words,
    definitions=definitions,
    model="bert-base-uncased",
    embedding_model=emb_model
)
```

This returns a transformers model and tokenizer with your new added vocabulary.
