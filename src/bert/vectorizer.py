"""BERT-based template vectorizer used in stage 3 of the hybrid pipeline."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer


class TemplateVectorizer:
    """Vectorize Drain templates into 768-dim BERT [CLS] embeddings.

    Wraps ``bert-base-uncased`` (or any BERT-family checkpoint) and extends the
    tokenizer with the placeholder tokens produced by the regex normalizer in
    stage 1 (``<IP>``, ``<NUM>``, ``<PATH>``, ``<HEX>``, ``<EXC>``,
    ``<SESSION>``, ``<UUID>``). The embeddings of the new tokens are
    initialized to the mean of the original vocabulary embeddings — random
    initialization would give placeholders no semantic anchor and degrade
    the resulting template vectors.

    The model is kept in ``eval`` mode and runs on CPU by default; this matches
    the project setup (no GPU, no fine-tuning, only pretrained weights with
    embedding extension).
    """

    PLACEHOLDER_TOKENS: list[str] = [
        "<IP>",
        "<NUM>",
        "<PATH>",
        "<HEX>",
        "<EXC>",
        "<SESSION>",
        "<UUID>",
    ]

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        placeholder_tokens: list[str] | None = None,
        device: str = "cpu",
        max_length: int = 128,
    ) -> None:
        """Load the model + tokenizer and inject mean-initialized placeholder tokens."""
        self.model_name = model_name
        self.placeholder_tokens = (
            list(placeholder_tokens) if placeholder_tokens is not None else list(self.PLACEHOLDER_TOKENS)
        )
        self.device = device
        self.max_length = max_length

        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        self.vocab_size_before: int = len(tokenizer)
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": self.placeholder_tokens}
        )
        self.vocab_size_after: int = len(tokenizer)
        self.num_added_tokens: int = num_added

        model.resize_token_embeddings(len(tokenizer))

        with torch.no_grad():
            embedding_layer = model.get_input_embeddings()
            original_size = embedding_layer.weight.size(0) - len(self.placeholder_tokens)
            mean_embedding = embedding_layer.weight[:original_size].mean(dim=0)
            for token in self.placeholder_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                embedding_layer.weight[token_id] = mean_embedding.clone()

        model.eval()
        model.to(device)

        self.tokenizer = tokenizer
        self.model = model
        self.new_token_ids: list[int] = [
            tokenizer.convert_tokens_to_ids(t) for t in self.placeholder_tokens
        ]

    def vectorize(self, template: str) -> np.ndarray:
        """Return the 768-dim [CLS] vector for a single template, dtype float32."""
        encoded = self.tokenizer(
            template,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        cls = outputs.last_hidden_state[0, 0, :]
        return cls.cpu().numpy().astype(np.float32)

    def vectorize_batch(
        self,
        templates: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Vectorize many templates in mini-batches, returning a ``(N, 768)`` float32 array."""
        if len(templates) == 0:
            return np.zeros((0, 768), dtype=np.float32)

        chunks: list[np.ndarray] = []
        indices: Iterable[int] = range(0, len(templates), batch_size)
        if show_progress:
            indices = tqdm(list(indices), desc="vectorize_batch")

        for start in indices:
            batch = templates[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(**encoded)
            elapsed = time.perf_counter() - t0
            if elapsed > 30.0:
                print(
                    f"WARNING: batch of {len(batch)} templates took {elapsed:.1f}s "
                    f"(>30s). Batching may not be working as expected."
                )

            cls = outputs.last_hidden_state[:, 0, :]
            chunks.append(cls.cpu().numpy().astype(np.float32))

        return np.concatenate(chunks, axis=0)

    def cosine_similarity(self, template_a: str, template_b: str) -> float:
        """Return cosine similarity between two template vectors, in ``[-1, 1]``."""
        a = self.vectorize(template_a)
        b = self.vectorize(template_b)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def get_tokenization(self, template: str) -> list[str]:
        """Return the WordPiece tokens produced by the (extended) tokenizer."""
        return self.tokenizer.tokenize(template)


if __name__ == "__main__":
    sample = "Send worker <NUM> failed"

    plain_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    plain_tokens = plain_tokenizer.tokenize(sample)
    print(f"Plain tokenization of {sample!r}: {plain_tokens}")
    assert "<NUM>" not in plain_tokens, (
        "Plain bert-base-uncased should fragment <NUM>; got it as a single token unexpectedly."
    )

    vectorizer = TemplateVectorizer()
    extended_tokens = vectorizer.get_tokenization(sample)
    print(f"Extended tokenization of {sample!r}: {extended_tokens}")
    assert "<NUM>" in extended_tokens, (
        "After extension, <NUM> must appear as a single token."
    )

    vec = vectorizer.vectorize(sample)
    assert vec.shape == (768,), f"Expected (768,), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"

    print("OK")
