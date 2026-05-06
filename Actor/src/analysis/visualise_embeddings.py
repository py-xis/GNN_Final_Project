"""
t-SNE visualisation of raw features and learned penultimate-layer embeddings
for all six Actor models.

Extracts embeddings via forward hooks (no model rewrite needed).
Uses split 0 only (representative; avoids multiplying compute).

Output: reports/milestone2/figures/tsne_grid.png
        reports/milestone2/figures/tsne_{model}.png (individual plots)

Run from project root:
    python src/analysis/visualise_embeddings.py \
        --data_root data/actor \
        --best_params_csv reports/milestone2/tables/best_params.csv \
        --out_dir reports/milestone2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.load_actor import load_actor
from src.hpo import build_model
from src.train import get_class_weights, train_model, evaluate
from src.utils.seed import set_seed


SPLIT_IDX  = 0
TSNE_PERP  = 30
TSNE_ITER  = 1000
CLASS_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
MODEL_TITLES = {
    "raw":      "Raw Features",
    "mlp":      "MLP",
    "gcn":      "GCN",
    "sage":     "GraphSAGE",
    "gat":      "GAT",
    "appnp":    "APPNP",
}


# ─── data ───────────────────────────────────────────────────────────────────

def preprocess_features(x):
    return (x.float() / x.sum(dim=1, keepdim=True).clamp(min=1e-6))


def load_best_params(csv_path):
    df = pd.read_csv(csv_path)
    params_by_model = {}
    for _, row in df.iterrows():
        d = row.dropna().to_dict()
        name = d.pop("model")
        for k in ["hidden_dim", "num_layers", "heads", "mlp_layers", "K"]:
            if k in d: d[k] = int(d[k])
        for k in ["alpha", "lr", "weight_decay", "dropout"]:
            if k in d: d[k] = float(d[k])
        params_by_model[name] = d
    return params_by_model


# ─── embedding extraction ────────────────────────────────────────────────────

def get_penultimate_embedding(model, model_name, x, edge_index):
    """
    Extract penultimate-layer representation via a forward hook.
    Returns numpy array of shape (N, hidden_dim).
    """
    embedding = {}

    def hook_fn(module, input, output):
        embedding["value"] = output.detach().cpu()

    # Register hook on the second-to-last module in the backbone
    if model_name == "mlp":
        # Layers: [Linear(in, h), Linear(h, h), ..., Linear(h, out)]
        # Hook on the last hidden Linear (index -2)
        handle = model.layers[-2].register_forward_hook(hook_fn)
    elif model_name in ("gcn", "sage"):
        handle = model.convs[-2].register_forward_hook(hook_fn)
    elif model_name == "gat":
        handle = model.convs[-2].register_forward_hook(hook_fn)
    elif model_name == "appnp":
        # Hook on the last Linear of the MLP part (before propagation)
        handle = model.lins[-1].register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    with torch.no_grad():
        _ = model(x, edge_index)
    handle.remove()

    return embedding["value"].numpy()


def train_and_embed(model_name, params, x, y, edge_index, data,
                    split_idx, num_classes, device):
    set_seed(42 + split_idx)
    train_mask = data.train_mask[:, split_idx].to(device)
    val_mask   = data.val_mask[:,   split_idx].to(device)
    test_mask  = data.test_mask[:,  split_idx].to(device)

    class_weights = get_class_weights(y, train_mask, num_classes, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    model     = build_model(model_name, params, x.shape[1], num_classes, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )

    train_model(model, x, y, edge_index, train_mask, val_mask,
                optimizer, criterion, max_epochs=1000, patience=100)

    # Per-node predictions for test set
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    preds = logits.argmax(dim=1).cpu().numpy()
    correct = (preds == y.cpu().numpy())

    embedding = get_penultimate_embedding(model, model_name,
                                          x.cpu(), edge_index.cpu())
    return embedding, preds, correct, test_mask.cpu().numpy().astype(bool)


# ─── t-SNE ──────────────────────────────────────────────────────────────────

def run_tsne(emb, seed=42):
    """PCA to 50 dims first (standard practice for high-d), then t-SNE."""
    n_components_pca = min(50, emb.shape[1] - 1)
    if emb.shape[1] > 50:
        pca = PCA(n_components=n_components_pca, random_state=seed)
        emb = pca.fit_transform(emb)
    tsne = TSNE(n_components=2, perplexity=TSNE_PERP,
                max_iter=TSNE_ITER, random_state=seed, init="pca")
    return tsne.fit_transform(emb)


# ─── plotting ────────────────────────────────────────────────────────────────

def plot_tsne_panel(ax, coords_2d, labels, test_mask, correct,
                    title, mark_errors=False):
    """Draw a single t-SNE panel on ax."""
    for c in sorted(np.unique(labels)):
        idx = labels == c
        ax.scatter(coords_2d[idx, 0], coords_2d[idx, 1],
                   s=4, alpha=0.5, color=CLASS_COLORS[c],
                   label=f"Class {c}", rasterized=True)

    if mark_errors:
        # Black ring on test nodes that were misclassified
        test_err = test_mask & (~correct)
        if test_err.sum() > 0:
            ax.scatter(coords_2d[test_err, 0], coords_2d[test_err, 1],
                       s=20, facecolors="none", edgecolors="black",
                       linewidths=0.5, alpha=0.6, label="Misclassified",
                       rasterized=True)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def save_individual(coords_2d, labels, test_mask, correct,
                    title, out_path, mark_errors=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_tsne_panel(ax, coords_2d, labels, test_mask, correct, title, mark_errors)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                           color=CLASS_COLORS[c], label=f"Class {c}", markersize=5)
               for c in range(5)]
    if mark_errors:
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                   markerfacecolor="none", markeredgecolor="black",
                                   label="Misclassified", markersize=5))
    ax.legend(handles=handles, loc="upper right", fontsize=7, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",       default="data/actor")
    parser.add_argument("--best_params_csv", default="reports/milestone2/tables/best_params.csv")
    parser.add_argument("--out_dir",         default="reports/milestone2")
    parser.add_argument("--device",          default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("Loading Actor dataset...")
    dataset = load_actor(root=args.data_root)
    data    = dataset[0]
    num_classes = dataset.num_classes

    x_raw = preprocess_features(data.x)
    y_all = data.y.numpy()
    x_dev = x_raw.to(device)
    y_dev = data.y.to(device)
    edge_index = data.edge_index.to(device)

    test_mask_0 = data.test_mask[:, SPLIT_IDX].numpy().astype(bool)
    params_by_model = load_best_params(args.best_params_csv)

    GRAPH_MODELS = ["mlp", "gcn", "sage", "gat", "appnp"]

    # ── Raw features t-SNE ───────────────────────────────────────────────────
    print("\nComputing t-SNE for raw features...")
    raw_emb   = x_raw.numpy()
    coords_raw = run_tsne(raw_emb)
    save_individual(coords_raw, y_all,
                    test_mask_0, np.ones(len(y_all), dtype=bool),
                    "Raw Features (t-SNE)",
                    os.path.join(fig_dir, "tsne_raw.png"), mark_errors=False)

    # ── Per-model embeddings ─────────────────────────────────────────────────
    all_coords  = {"raw": coords_raw}
    all_correct = {"raw": np.ones(len(y_all), dtype=bool)}

    for model_name in GRAPH_MODELS:
        print(f"\nTraining {model_name} and extracting embedding (split {SPLIT_IDX})...")
        params = params_by_model[model_name]
        embedding, preds, correct, _ = train_and_embed(
            model_name, params, x_dev, y_dev, edge_index,
            data, SPLIT_IDX, num_classes, device
        )
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Running t-SNE...")
        coords = run_tsne(embedding)
        all_coords[model_name]  = coords
        all_correct[model_name] = correct

        # Mark errors on MLP and GCN only
        mark = model_name in ("mlp", "gcn")
        save_individual(
            coords, y_all, test_mask_0, correct,
            f"{MODEL_TITLES[model_name]} Embedding (t-SNE)",
            os.path.join(fig_dir, f"tsne_{model_name}.png"),
            mark_errors=mark
        )

    # ── 2×3 grid ─────────────────────────────────────────────────────────────
    print("\nSaving 2×3 t-SNE grid...")
    panel_order = ["raw", "mlp", "gcn", "sage", "gat", "appnp"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for i, key in enumerate(panel_order):
        mark = key in ("mlp", "gcn")
        plot_tsne_panel(axes_flat[i], all_coords[key], y_all,
                        test_mask_0, all_correct[key],
                        MODEL_TITLES.get(key, key), mark_errors=mark)

    # Shared legend below
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                           color=CLASS_COLORS[c], label=f"Class {c}", markersize=6)
               for c in range(5)]
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                               markerfacecolor="none", markeredgecolor="black",
                               label="Misclassified (MLP/GCN only)", markersize=6))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Actor — t-SNE of Features and Learned Embeddings\n"
                 "(split 0, coloured by true class)", fontsize=12, y=1.01)
    fig.tight_layout()
    grid_path = os.path.join(fig_dir, "tsne_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {grid_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
