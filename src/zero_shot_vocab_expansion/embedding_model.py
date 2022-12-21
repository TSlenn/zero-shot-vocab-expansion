from sentence_transformers import SentenceTransformer, models


class EmbeddingModel(SentenceTransformer):
    def __init__(
        self,
        backbone_model: str,
        max_seq_length: int = 512,
        dense_layers: int = 3,
        freeze_backbone: bool = False
    ):
        """Builds sentence-transformers model for word definition embedding.

        Args:
            backbone_model (str): Name or path to pretrained backbone model.
            max_seq_length (int, optional): Max token length for model input.
                Defaults to 256.
            dense_layers (int, optional): Number of dense layers in the
                projection head. Defaults to 3.

        Returns:
            SentenceTransformer: Huggingface SentenceTransformer model with
                backbone, sentence pooling, and dense projection head.
        """
        backbone = models.Transformer(
            backbone_model, max_seq_length=max_seq_length
        )
        ft_dim = backbone.get_word_embedding_dimension()
        pooler = models.Pooling(ft_dim)
        modules = [backbone, pooler]
        # projection head
        for _ in range(dense_layers):
            modules.append(
                models.Dense(in_features=ft_dim, out_features=ft_dim)
            )
        super().__init__(modules=modules)
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freezes the weights in the transformer backbone."""
        auto_model = self._first_module().auto_model
        for param in auto_model.parameters():
            param.requires_grad = False
