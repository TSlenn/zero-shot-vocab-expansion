from torch.utils.data import Dataset
import random
from tqdm.contrib.concurrent import thread_map
import multiprocessing as mp
from .utils import get_definitions


class VocabDataset(Dataset):
    def __init__(self, model, tokenizer):
        self.embeddings = model.embeddings.word_embeddings.weight.detach()
        self.token_map = tokenizer.vocab
        words = list(self.token_map.keys())
        definitions = thread_map(
            get_definitions, words, max_workers=mp.cpu_count()
        )
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
