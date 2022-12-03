from sentence_transformers import SentenceTransformer, models


def embedding_model(
    base_model,
    max_seq_length: int = 256,
    dense_layers: int = 3
):
    """Builds sentence-transformers model for word embedding initialization.

    Args:
        base_model (str): Name or path to pretrained backbone model.
        max_seq_length (int, optional): Max token length for model input.
            Defaults to 256.
        dense_layers (int, optional): Number of dense layers in the
            projection head. Defaults to 3.

    Returns:
        SentenceTransformer: Huggingface SentenceTransformer model with
            base model backbone, sentence pooling, and dense projection head.
    """
    backbone = models.Transformer(base_model, max_seq_length=max_seq_length)
    ft_dim = backbone.get_word_embedding_dimension()
    pooler = models.Pooling(ft_dim)
    # projection head
    dense1 = models.Dense(in_features=ft_dim, out_features=ft_dim)
    dense2 = models.Dense(in_features=ft_dim, out_features=ft_dim)
    dense3 = models.Dense(in_features=ft_dim, out_features=ft_dim)
    model = SentenceTransformer(
        modules=[backbone, pooler, dense1, dense2, dense3]
    )

    return model
