from zero_shot_vocab_expansion.evaluation import MSEDefinitionEvaluator
from zero_shot_vocab_expansion import EmbeddingModel
import numpy as np


def test_MSEDefinitionEvaluator():
    model = EmbeddingModel("bert-base-uncased")
    definitions = [
        ["Test definition one.", "Test definition 2"],
        ["Test definition second word."]
    ]
    embeddings = np.random.normal(size=(2, 768))
    x = MSEDefinitionEvaluator(definitions, embeddings)
    x(model, output_path="")
