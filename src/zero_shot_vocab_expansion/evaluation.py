from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
import logging
import os
import csv
from typing import List

logger = logging.getLogger(__name__)


class MSEDefinitionEvaluator(SentenceEvaluator):
    """
    Computes the mean squared error (x100) between the computed sentence
    embedding and some target sentence embedding. The MSE is computed between
    ||target_embeddings - model.encode(target_sentences)||.
    This is a copy of MSEEvaluator, with 'teacher_model' replaced by hardcoded
    embeddings.

    During evaluation, words with multiple definitions have each definition
    encoded, then the embeddings are averaged. This differs from training loss,
    which only uses one definition.
    """
    def __init__(
        self,
        definitions: List[List[str]],
        embeddings: np.ndarray,
        name: str = '',
        write_csv: bool = False
    ):
        self.embeddings = embeddings
        self.definitions = definitions
        self.name = name

        self.csv_file = "mse_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MSE"]
        self.write_csv = write_csv

    def __call__(self, model, output_path, epoch=-1, steps=-1):
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        # Average embeddings of all definitions for each word
        target_embeddings = np.stack(
            [np.mean(model.encode(x), axis=0) for x in self.definitions]
        )

        mse = ((self.embeddings - target_embeddings)**2).mean()
        mse *= 100

        logger.info(
            "MSE evaluation (lower = better) on "+self.name+" dataset"+out_txt
        )
        logger.info("MSE (*100):\t{:4f}".format(mse))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f: # noqa
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mse])

        # Return negative score, SentenceTransformers maximizes the performance
        return -mse
