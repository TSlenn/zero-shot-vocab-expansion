## Initial Setup

```
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
to a custom model.

```
from zero_shot_vocab_expansion import EmbeddingModel


model = EmbeddingModel(backbone_model = "bert-base-uncased")
```

EmbeddingModel has 3 additional optional args:
- max_seq_length: Maximum token length for the model. Defaults to 256.
- dense_layers: Number of dense layers in the projection head.
- freeze_backbone: If True, freezes the weights in the transformer model.
    Defaults to False.

freeze_backbone is useful if you have limited data or limited compute.  
If you'd like to freeze the backbone after initializing the model:

```
model._freeze_backbone()
```
