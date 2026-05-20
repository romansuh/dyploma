"""Last-layer [CLS]->token attention extraction from pretrained BERT.

Used by notebook 05 to render attention bars for FP/FN templates and for
templates where the three classifiers disagree (thesis 1.6, 2.3 Перевірка 3).
The function is pure: it does not mutate model/tokenizer state and does not
fine-tune. Attention comes from the pretrained checkpoint as loaded by
src/bert/vectorizer.py.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import BertModel, BertTokenizer


def extract_cls_attention(
    model: BertModel,
    tokenizer: BertTokenizer,
    template: str,
    max_length: int = 128,
    device: str = "cpu",
) -> tuple[list[str], np.ndarray]:
    """Forward `template` through BERT, return ([tokens], [CLS]->token attention).

    Procedure (shapes annotated inline):
        encoded.input_ids       (1, S)
        outputs.attentions      tuple of L tensors, each (1, H, S, S)
        last_layer              (1, H, S, S)  # outputs.attentions[-1]
        head_mean               (S, S)        # mean over H, drop batch dim
        cls_row                 (S,)          # row 0 of head_mean = [CLS] -> token

    Args:
        model: BertModel in eval mode (callable with output_attentions=True at call
            time, so no config edit is required).
        tokenizer: matching BertTokenizer; expected to be the same instance that
            had the 7 placeholders (<IP>, <NUM>, <PATH>, <HEX>, <EXC>, <SESSION>,
            <UUID>) added by `TemplateVectorizer`. Required so subword tokens
            agree with the embeddings the model was vectorized with.
        template: raw template string with placeholders preserved.
        max_length: tokenizer truncation length (default 128, matches nb02).
        device: 'cpu' per project constraint.

    Returns:
        tokens: list[str] of length S (decoded subwords including [CLS], [SEP]).
        attention: float32 ndarray shape (S,) — [CLS] attention to each token,
            averaged across the H heads of the final encoder layer.
    """
    encoded = tokenizer(
        template,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    # outputs.attentions: tuple length L, each tensor (1, H, S, S).
    last_layer = outputs.attentions[-1]                  # (1, H, S, S)
    head_mean = last_layer.mean(dim=1)[0]                # (S, S)
    cls_row = head_mean[0]                               # (S,)

    input_ids = encoded["input_ids"][0]                  # (S,)
    tokens: list[str] = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    attention: np.ndarray = cls_row.cpu().numpy().astype(np.float32)
    assert len(tokens) == attention.size, (
        f"token/attention length mismatch: {len(tokens)} vs {attention.size}"
    )
    return tokens, attention


if __name__ == "__main__":
    # Smoke test: load tokenizer through TemplateVectorizer (so the 7 placeholder
    # tokens are added and embeddings are mean-initialized), then reload BertModel
    # with attn_implementation="eager". The default SDPA backend in transformers 5.x
    # does not expose attention weights — the eager backend does. Embeddings are
    # copied from the vectorizer's model so placeholder tokens map to the same
    # mean-initialized vectors that nb02 used for embedding.
    import torch
    from transformers import BertModel

    from src.bert.vectorizer import TemplateVectorizer

    vec = TemplateVectorizer(device="cpu")
    eager_model = BertModel.from_pretrained(
        "bert-base-uncased", attn_implementation="eager"
    )
    eager_model.resize_token_embeddings(len(vec.tokenizer))
    with torch.no_grad():
        eager_model.embeddings.word_embeddings.weight.copy_(
            vec.model.embeddings.word_embeddings.weight
        )
    eager_model.eval()

    tokens, att = extract_cls_attention(
        eager_model, vec.tokenizer, "Got zxid <HEX> expected <HEX>"
    )
    print(f"tokens ({len(tokens)}): {tokens}")
    print(f"attention shape={att.shape} sum={att.sum():.4f} max={att.max():.4f}")
    assert att.shape == (len(tokens),)
    assert abs(att.sum() - 1.0) < 1e-3, (
        f"row of an attention matrix should sum to ~1, got {att.sum():.4f}"
    )
    print("OK")
