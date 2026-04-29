from __future__ import annotations

import math
import random
from collections import Counter
from types import SimpleNamespace

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB


_tokenizer = get_tokenizer("basic_english")


class VocabCompat:
    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {token: idx for idx, token in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)


class FieldCompat:
    def __init__(self, sequential: bool = True, lower: bool = False, fix_length: int | None = None, batch_first: bool = False) -> None:
        self.sequential = sequential
        self.lower = lower
        self.fix_length = fix_length
        self.batch_first = batch_first
        self.unk_token = "<unk>"
        self.pad_token = "<pad>" if sequential else None
        self.vocab: VocabCompat | None = None

    def preprocess(self, value):
        if not self.sequential:
            return value.lower() if self.lower and isinstance(value, str) else value

        tokens = _tokenizer(value) if isinstance(value, str) else list(value)
        if self.lower:
            tokens = [token.lower() for token in tokens]
        return tokens

    def build_vocab(self, dataset, max_size: int | None = None, min_freq: int = 1, vectors=None) -> None:
        del vectors
        counter: Counter[str] = Counter()
        for example in dataset.examples:
            counter.update(example.text)

        tokens = [token for token, freq in counter.most_common() if freq >= min_freq]
        if max_size is not None:
            tokens = tokens[:max_size]

        ordered_tokens = []
        for token in [self.unk_token, self.pad_token, *tokens]:
            if token is not None and token not in ordered_tokens:
                ordered_tokens.append(token)
        self.vocab = VocabCompat(ordered_tokens)

    def process(self, batch_tokens: list[list[str]]) -> torch.Tensor:
        if self.vocab is None:
            raise RuntimeError("build_vocab must be called before process")

        pad_id = self.vocab.stoi[self.pad_token]
        unk_id = self.vocab.stoi[self.unk_token]

        prepared_tokens = []
        for tokens in batch_tokens:
            tokens = list(tokens)
            if self.fix_length is not None:
                tokens = tokens[: self.fix_length]
            prepared_tokens.append(tokens)

        max_len = self.fix_length or max(1, max(len(tokens) for tokens in prepared_tokens))
        batch_ids = []
        for tokens in prepared_tokens:
            padded = tokens + [self.pad_token] * (max_len - len(tokens))
            batch_ids.append([self.vocab.stoi.get(token, unk_id) for token in padded])

        tensor = torch.tensor(batch_ids, dtype=torch.long)
        return tensor if self.batch_first else tensor.t().contiguous()


class LabelFieldCompat(FieldCompat):
    def __init__(self, batch_first: bool = False) -> None:
        super().__init__(sequential=False, lower=False, batch_first=batch_first)

    def build_vocab(self, dataset, max_size: int | None = None, min_freq: int = 1, vectors=None) -> None:
        del max_size, min_freq, vectors
        ordered_labels = ["<unk>"]
        for example in dataset.examples:
            label = example.label
            if label not in ordered_labels:
                ordered_labels.append(label)
        self.vocab = VocabCompat(ordered_labels)

    def process(self, labels: list[str]) -> torch.Tensor:
        if self.vocab is None:
            raise RuntimeError("build_vocab must be called before process")
        return torch.tensor([self.vocab.stoi.get(label, 0) for label in labels], dtype=torch.long)


class DatasetCompat:
    def __init__(self, examples, text_field: FieldCompat, label_field: LabelFieldCompat) -> None:
        self.examples = examples
        self.text_field = text_field
        self.label_field = label_field

    def __len__(self) -> int:
        return len(self.examples)

    def split(self, split_ratio: float = 0.8, random_state=None):
        indices = list(range(len(self.examples)))
        rng = random_state if isinstance(random_state, random.Random) else random.Random(0)
        rng.shuffle(indices)
        cut = int(len(indices) * split_ratio)
        left = [self.examples[idx] for idx in indices[:cut]]
        right = [self.examples[idx] for idx in indices[cut:]]
        return (
            DatasetCompat(left, self.text_field, self.label_field),
            DatasetCompat(right, self.text_field, self.label_field),
        )


class IteratorCompat:
    def __init__(self, dataset: DatasetCompat, batch_size: int, device, shuffle: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset.examples)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_examples = [self.dataset.examples[idx] for idx in batch_indices]
            batch_text = self.dataset.text_field.process([example.text for example in batch_examples]).to(self.device)
            batch_label = self.dataset.label_field.process([example.label for example in batch_examples]).to(self.device)
            yield SimpleNamespace(text=batch_text, label=batch_label)


def load_imdb_splits(text_field: FieldCompat, label_field: LabelFieldCompat, root: str = ".data"):
    train_iter, test_iter = IMDB(root=root, split=("train", "test"))

    def to_examples(data_iter):
        return [
            SimpleNamespace(
                text=text_field.preprocess(text),
                label=label_field.preprocess(label),
            )
            for label, text in data_iter
        ]

    return (
        DatasetCompat(to_examples(train_iter), text_field, label_field),
        DatasetCompat(to_examples(test_iter), text_field, label_field),
    )


def make_bucket_iterators(datasets, batch_size: int, device):
    return tuple(
        IteratorCompat(dataset, batch_size=batch_size, device=device, shuffle=index == 0)
        for index, dataset in enumerate(datasets)
    )
