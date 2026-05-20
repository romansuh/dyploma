"""Generate 5 PDF figures for thesis appendices.

Outputs (./thesis_figures/):
    tsne_templates.pdf, umap_templates.pdf,
    attn_t5_fp.pdf, attn_t62_fn.pdf, attn_t63_fn.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import umap
from matplotlib import colormaps
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from transformers import BertModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bert.vectorizer import TemplateVectorizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)

EMB_PATH = ROOT / "data/processed/embeddings/zookeeper_full_embeddings.npy"
IDMAP_PATH = ROOT / "data/processed/embeddings/zookeeper_full_id_mapping.json"
LABELS_PATH = ROOT / "data/processed/clusters/template_to_binary_label.json"
TOKENIZER_DIR = ROOT / "data/processed/tokenizer"

COLOR_NEG = "#1f77b4"
COLOR_POS = "#ff7f0e"


def load_vectors_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """Return (L2-normalized 77×768 vectors, binary labels in row order)."""
    vectors = np.load(EMB_PATH)
    vectors = normalize(vectors, norm="l2", axis=1)

    with IDMAP_PATH.open() as f:
        idmap = json.load(f)
    with LABELS_PATH.open() as f:
        records = json.load(f)
    tid_to_label = {r["template_id"]: int(r["final_label"]) for r in records}

    labels = np.array(
        [tid_to_label[entry["template_id"]] for entry in idmap], dtype=np.int64
    )
    assert vectors.shape == (77, 768), vectors.shape
    assert labels.shape == (77,), labels.shape
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    assert n_pos == 16 and n_neg == 61, (n_pos, n_neg)
    return vectors, labels


def scatter_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Two-class scatter with the styling required by the thesis."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for label_value, color, legend in (
        (0, COLOR_NEG, "ℓ = 0 (неінформативні, n=61)"),
        (1, COLOR_POS, "ℓ = 1 (інформативні, n=16)"),
    ):
        mask = labels == label_value
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=60,
            alpha=0.85,
            c=color,
            edgecolors="black",
            linewidths=0.5,
            label=legend,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_tsne(vectors: np.ndarray, labels: np.ndarray) -> None:
    tsne = TSNE(
        n_components=2,
        perplexity=10,
        random_state=42,
        metric="cosine",
        init="pca",
    )
    coords = tsne.fit_transform(vectors)
    scatter_2d(
        coords,
        labels,
        xlabel="t-SNE dim 1",
        ylabel="t-SNE dim 2",
        out_path=OUT_DIR / "tsne_templates.pdf",
    )


def make_umap(vectors: np.ndarray, labels: np.ndarray) -> None:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=8,
        min_dist=0.1,
        random_state=42,
        metric="cosine",
    )
    coords = reducer.fit_transform(vectors)
    scatter_2d(
        coords,
        labels,
        xlabel="UMAP dim 1",
        ylabel="UMAP dim 2",
        out_path=OUT_DIR / "umap_templates.pdf",
    )


def build_attention_model(vectorizer: TemplateVectorizer) -> BertModel:
    """Eager-attention BERT with placeholder embeddings copied from the vectorizer.

    The SDPA backend used by `TemplateVectorizer` silently drops attentions, so
    we load a second model with `attn_implementation='eager'` and copy the
    mean-initialized placeholder embeddings over to keep the forward identical.
    """
    model = BertModel.from_pretrained(
        vectorizer.model_name, attn_implementation="eager"
    )
    model.resize_token_embeddings(len(vectorizer.tokenizer))
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(
            vectorizer.model.get_input_embeddings().weight
        )
    model.eval()
    return model


def attention_cls_row(
    text: str,
    vectorizer: TemplateVectorizer,
    model: BertModel,
) -> tuple[list[str], np.ndarray]:
    """Return (token labels including [CLS]/[SEP], [CLS]-row attention vector)."""
    tokenizer = vectorizer.tokenizer
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    last_layer = out.attentions[-1][0]  # (heads, seq, seq)
    head_avg = last_layer.mean(dim=0)  # (seq, seq)
    cls_row = head_avg[0].cpu().numpy()  # (seq,)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    return tokens, cls_row


def make_attention_figure(
    text: str,
    vectorizer: TemplateVectorizer,
    model: BertModel,
    out_path: Path,
) -> None:
    tokens, weights = attention_cls_row(text, vectorizer, model)

    cmap = colormaps["viridis"]
    norm = Normalize(vmin=float(weights.min()), vmax=float(weights.max()))
    colors = [cmap(norm(w)) for w in weights]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(tokens)), weights, color=colors, edgecolor="black", linewidth=0.4)

    top3_idx = np.argsort(weights)[-3:]
    y_offset = weights.max() * 0.02
    for idx in top3_idx:
        ax.text(
            idx,
            weights[idx] + y_offset,
            f"{weights[idx]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontfamily="monospace")
    ax.set_xlabel("Субтокени шаблону")
    ax.set_ylabel("Вага уваги [CLS]")
    ax.set_ylim(0, weights.max() * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_template_text(template_id: int) -> str:
    with (ROOT / "data/processed/templates/zookeeper_full_templates.json").open() as f:
        templates = json.load(f)
    for entry in templates:
        if entry["id"] == template_id:
            return entry["template"]
    raise KeyError(template_id)


def main() -> None:
    vectors, labels = load_vectors_and_labels()
    print(f"Loaded {vectors.shape} vectors; labels: {int((labels==1).sum())}/+ {int((labels==0).sum())}/-")

    print("→ t-SNE …")
    make_tsne(vectors, labels)
    print("→ UMAP …")
    make_umap(vectors, labels)

    print("→ Attention figures …")
    vectorizer = TemplateVectorizer()
    attn_model = build_attention_model(vectorizer)

    for tid, suffix in ((5, "fp"), (62, "fn"), (63, "fn")):
        text = get_template_text(tid)
        print(f"   t{tid}: {text!r}")
        make_attention_figure(
            text,
            vectorizer,
            attn_model,
            OUT_DIR / f"attn_t{tid}_{suffix}.pdf",
        )

    print(f"\nFigures written to {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.pdf")):
        print("  ", p.name, f"{p.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
