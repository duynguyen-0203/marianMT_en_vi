from datasets import load_metric
import string
from typing import List

import torch


class Evaluator:
    def __init__(self, dataset, tokenizer):
        self._metric = load_metric('bleu')
        self._dataset = dataset
        self._tokenizer = tokenizer

        self.predictions = []
        self.references = []

        self._convert_references()

    def eval_batch(self, outputs: torch.tensor):
        for output in outputs:
            tokens = self._tokenizer.convert_ids_to_tokens(output, skip_special_tokens=True)
            self.predictions.append(convert_tokens(tokens))

    def compute_scores(self):
        assert len(self.predictions) == len(self.references)
        results = self._metric.compute(predictions=self.predictions, references=self.references)

        return results

    def _convert_references(self):
        for sample in self._dataset.samples:
            self.references.append([sample.target.split()])


def convert_tokens(tokens: List[str]):
    res = []
    for i, token in enumerate(tokens):
        if token.startswith(chr(9601)):
            res.append(token[1:])
        elif token in string.punctuation:
            res.append(token)
        elif len(res) != 0:
            res[-1] += token
        else:
            res.append(token)

    return res
