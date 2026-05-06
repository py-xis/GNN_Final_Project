"""
Depth-wise ablation experiments for Actor.

This script sweeps message-passing depth for GCN, GraphSAGE, and GAT:
    num_layers in {2, 3, 4, 5}

All runs use the best HPO parameters except for num_layers, evaluate on the
first 3 predefined Actor splits, and use the same training loop as the main
experiment.

Outputs:
    reports/milestone2/tables/ablation_depth_sweep.csv
    reports/milestone2/figures/ablation_depth_sweep.png

Run from project root:
    python src/analysis/ablations.py \
        --data_root data/actor \
        --best_params_csv reports/milestone2/tables/best_params.csv \
        --out_dir reports/milestone2
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = os.path.join("/tmp", "matplotlib")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_config_dir

if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = os.path.join("/tmp", "fontconfig-cache")
    os.makedirs(xdg_cache_dir, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = xdg_cache_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.load_actor import load_actor
from src.hpo import build_model
from src.train import get_class_weights, train_model, evaluate
from src.utils.seed import set_seed


ABLATION_SPLITS = [0, 1, 2]
DEPTH_MODELS = ["gcn", "sage", "gat"]
DEPTHS = [2, 3, 4, 5]
MODEL_COLORS = {
    "gcn": "#e15759",
    "sage": "#76b7b2",
    "gat": "#59a14f",
}


def preprocess_features(x):
    row_sums = x.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return x.float() / row_sums


def load_best_params(best_params_csv):
    df = pd.read_csv(best_params_csv)
    params_by_model = {}
    for _, row in df.iterrows():
        params = row.dropna().to_dict()
        model_name = params.pop("model")
        for key in ["hidden_dim", "num_layers", "heads", "mlp_layers", "K"]:
            if key in params:
                params[key] = int(params[key])
        for key in ["alpha", "lr", "weight_decay", "dropout"]:
            if key in params:
                params[key] = float(params[key])
        params_by_model[model_name] = params
    return params_by_model


def run_one(model_name, params, x, y, edge_index,
            data, split_idx, num_classes, device):
    set_seed(42 + split_idx)
    train_mask = data.train_mask[:, split_idx].to(device)
    val_mask = data.val_mask[:, split_idx].to(device)
    test_mask = data.test_mask[:, split_idx].to(device)

    class_weights = get_class_weights(y, train_mask, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(model_name, params, x.shape[1], num_classes, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )

    train_model(model, x, y, edge_index, train_mask, val_mask,
                optimizer, criterion, max_epochs=1000, patience=100)

    _, acc, macro_f1, _, _ = evaluate(model, x, y, edge_index, test_mask, criterion)
    return acc, macro_f1


def mean_std_over_splits(model_name, params, x, y, edge_index,
                         data, splits, num_classes, device, tag):
    accs, f1s = [], []
    for split_idx in splits:
        acc, f1 = run_one(model_name, params, x, y, edge_index,
                          data, split_idx, num_classes, device)
        accs.append(acc)
        f1s.append(f1)
        print(f"    [{tag}] split {split_idx}: acc={acc:.4f}  macro-F1={f1:.4f}")
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def ablation_depth_sweep(x, y, edge_index, data, params_by_model,
                         num_classes, device, splits):
    rows = []
    for model_name in DEPTH_MODELS:
        base_params = params_by_model[model_name].copy()
        for depth in DEPTHS:
            params = {**base_params, "num_layers": depth}
            print(f"\n  [depth sweep] {model_name} depth={depth}")
            acc_m, acc_s, f1_m, f1_s = mean_std_over_splits(
                model_name, params, x, y, edge_index,
                data, splits, num_classes, device,
                tag=f"{model_name}-d{depth}",
            )
            rows.append({
                "model": model_name,
                "num_layers": depth,
                "macro_f1_mean": round(f1_m, 4),
                "macro_f1_std": round(f1_s, 4),
                "acc_mean": round(acc_m, 4),
                "acc_std": round(acc_s, 4),
            })
    return pd.DataFrame(rows)


def plot_depth_sweep(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name in DEPTH_MODELS:
        sub = df[df["model"] == model_name].sort_values("num_layers")
        color = MODEL_COLORS.get(model_name, "gray")
        ax.plot(
            sub["num_layers"], sub["macro_f1_mean"],
            marker="o", label=model_name.upper(), color=color, linewidth=2,
        )
        ax.fill_between(
            sub["num_layers"],
            sub["macro_f1_mean"] - sub["macro_f1_std"],
            sub["macro_f1_mean"] + sub["macro_f1_std"],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Depth Sweep on Actor")
    ax.set_xticks(DEPTHS)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/actor")
    parser.add_argument("--best_params_csv", default="reports/milestone2/tables/best_params.csv")
    parser.add_argument("--out_dir", default="reports/milestone2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    fig_dir = os.path.join(args.out_dir, "figures")
    table_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    print("Loading Actor dataset...")
    dataset = load_actor(root=args.data_root)
    data = dataset[0]
    num_classes = dataset.num_classes

    x = preprocess_features(data.x).to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    params_by_model = load_best_params(args.best_params_csv)

    print("\n" + "=" * 60)
    print("DEPTH-WISE ABLATION: GCN, GraphSAGE, GAT")
    print("=" * 60)
    t0 = time.time()

    df_depth = ablation_depth_sweep(
        x, y, edge_index, data, params_by_model,
        num_classes, device, ABLATION_SPLITS,
    )
    print(f"\nDepth sweep results:\n{df_depth.to_string(index=False)}")

    csv_path = os.path.join(table_dir, "ablation_depth_sweep.csv")
    fig_path = os.path.join(fig_dir, "ablation_depth_sweep.png")
    df_depth.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")
    plot_depth_sweep(df_depth, fig_path)

    print(f"\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
