"""
Milestone 2 – Fair comparison of LR, MLP, GCN, GraphSAGE, APPNP on Actor.

Run:
    python src/run_milestone2.py --data_root data/actor \
        --out_dir reports/milestone2 --seed 42 --num_trials 30
"""

import argparse
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, balanced_accuracy_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.load_actor import load_actor
from src.utils.seed import set_seed, get_device
from src.hpo import run_hpo, build_model
from src.train import get_class_weights, train_model, evaluate


ALL_MODELS = ["lr_model", "mlp", "gcn", "sage", "gat", "appnp"]
NUM_SPLITS = 10


class _TeeLogger:
    """Writes to both the original stream and a log file simultaneously."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def preprocess_features(x):
    """Row-normalise: divide each node's feature vector by its L1 norm."""
    row_sums = x.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return x.float() / row_sums


# ---------------------------------------------------------------------------
# Single-split evaluation with given params
# ---------------------------------------------------------------------------

def run_one_split(model_name, params, x, y, edge_index,
                  train_mask, val_mask, test_mask,
                  num_classes, device, split_idx):
    set_seed(42 + split_idx)

    class_weights = get_class_weights(y, train_mask, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(model_name, params, x.shape[1], num_classes, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )

    val_f1, actual_epochs = train_model(
        model, x, y, edge_index, train_mask, val_mask,
        optimizer, criterion, max_epochs=1000, patience=100
    )

    _, test_acc, test_macro_f1, test_preds, test_labels = evaluate(
        model, x, y, edge_index, test_mask, criterion
    )

    class_labels = list(range(num_classes))
    test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds, labels=class_labels)
    report = classification_report(test_labels, test_preds, labels=class_labels,
                                    zero_division=0, output_dict=True)

    print(f"    split {split_idx} | acc: {test_acc:.4f} | macro-F1: {test_macro_f1:.4f} "
          f"| weighted-F1: {test_weighted_f1:.4f} | epochs: {actual_epochs}")

    return {
        "split": split_idx,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "bal_acc": bal_acc,
        "epochs": actual_epochs,
        "confusion_matrix": cm,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_confusion_matrix(cm_sum, model_name, num_classes, out_dir):
    """Plot and save a normalised confusion matrix (sum over all splits)."""
    cm_norm = cm_sum.astype(float) / cm_sum.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=range(num_classes),
                yticklabels=range(num_classes), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {model_name} (normalised, summed over splits)")
    path = os.path.join(out_dir, "figures", f"confusion_matrix_{model_name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


def save_results(all_results, num_classes, out_dir):
    """Save summary table, per-class table, best params, and confusion matrices."""
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    summary_rows = []
    param_rows = []
    per_class_rows = []

    for model_name, result in all_results.items():
        splits = result["splits"]

        accs   = [s["test_acc"]         for s in splits]
        mf1s   = [s["test_macro_f1"]    for s in splits]
        wf1s   = [s["test_weighted_f1"] for s in splits]
        bals   = [s["bal_acc"]          for s in splits]
        epochs = [s["epochs"]           for s in splits]

        summary_rows.append({
            "model":            model_name,
            "acc_mean":         round(np.mean(accs),  4),
            "acc_std":          round(np.std(accs),   4),
            "macro_f1_mean":    round(np.mean(mf1s),  4),
            "macro_f1_std":     round(np.std(mf1s),   4),
            "weighted_f1_mean": round(np.mean(wf1s),  4),
            "weighted_f1_std":  round(np.std(wf1s),   4),
            "bal_acc_mean":     round(np.mean(bals),  4),
            "avg_epochs":       round(np.mean(epochs), 1),
        })

        param_rows.append({"model": model_name, **result["best_params"]})

        # Per-class F1 averaged over splits
        for c in range(num_classes):
            key = str(c)
            class_f1s = [s["report"][key]["f1-score"] for s in splits if key in s["report"]]
            class_rec  = [s["report"][key]["recall"]   for s in splits if key in s["report"]]
            per_class_rows.append({
                "model":        model_name,
                "class":        c,
                "f1_mean":      round(np.mean(class_f1s), 4),
                "f1_std":       round(np.std(class_f1s),  4),
                "recall_mean":  round(np.mean(class_rec),  4),
            })

        # Confusion matrix summed over splits
        cm_sum = np.sum([s["confusion_matrix"] for s in splits], axis=0)
        save_confusion_matrix(cm_sum, model_name, num_classes, out_dir)

    # Save CSVs
    df_summary = pd.DataFrame(summary_rows)
    df_params  = pd.DataFrame(param_rows)
    df_class   = pd.DataFrame(per_class_rows)

    df_summary.to_csv(os.path.join(out_dir, "tables", "results_summary.csv"),  index=False)
    df_params.to_csv( os.path.join(out_dir, "tables", "best_params.csv"),       index=False)
    df_class.to_csv(  os.path.join(out_dir, "tables", "per_class_f1.csv"),      index=False)

    print(f"\n  [saved] {out_dir}/tables/results_summary.csv")
    print(f"  [saved] {out_dir}/tables/best_params.csv")
    print(f"  [saved] {out_dir}/tables/per_class_f1.csv")

    return df_summary, df_class


def print_summary_table(df_summary):
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY  (mean ± std over 10 splits)")
    print("=" * 80)
    for _, row in df_summary.iterrows():
        print(f"  {row['model']:<10} | "
              f"acc: {row['acc_mean']:.4f}±{row['acc_std']:.4f} | "
              f"macro-F1: {row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} | "
              f"weighted-F1: {row['weighted_f1_mean']:.4f}±{row['weighted_f1_std']:.4f}")
    print("=" * 80)


def save_macro_f1_plot(df_summary, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    models = df_summary["model"]
    means  = df_summary["macro_f1_mean"]
    stds   = df_summary["macro_f1_std"]
    ax.bar(models, means, yerr=stds, capsize=5,
           color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Actor – Test Macro-F1 by Model (mean ± std, 10 splits)")
    ax.set_ylim(0, max(means) * 1.2)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)
    path = os.path.join(out_dir, "figures", "macro_f1_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


def save_per_class_f1_plot(df_class, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    models = df_class["model"].unique()
    classes = sorted(df_class["class"].unique())
    x = np.arange(len(classes))
    width = 0.15

    for i, model in enumerate(models):
        sub = df_class[df_class["model"] == model].sort_values("class")
        offset = (i - len(models) / 2) * width + width / 2
        ax.bar(x + offset, sub["f1_mean"], width,
               label=model, edgecolor="black", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.set_ylabel("Mean F1")
    ax.set_title("Actor – Per-Class F1 by Model (mean over 10 splits)")
    ax.legend()
    path = os.path.join(out_dir, "figures", "per_class_f1.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str,  default="data/actor")
    parser.add_argument("--out_dir",    type=str,  default="reports/milestone2")
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--num_trials", type=int,  default=30,
                        help="Number of HPO trials per model")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        choices=ALL_MODELS,
                        help="Subset of models to run. Default: all of them.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tables"),  exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "logs"),    exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, "logs", f"run_{timestamp}.log")
    _log_file = open(log_path, "w", buffering=1)
    sys.stdout = _TeeLogger(sys.stdout, _log_file)
    print(f"Logging to: {log_path}")
    print(f"Run started: {datetime.datetime.now().isoformat()}")

    # Load and preprocess data
    print("\nLoading Actor dataset...")
    dataset = load_actor(root=args.data_root)
    data = dataset[0]
    num_classes = dataset.num_classes

    x = preprocess_features(data.x).to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)

    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, "
          f"Features: {data.num_node_features}, Classes: {num_classes}")

    models_to_run = args.models if args.models else ALL_MODELS
    print(f"Models to run: {models_to_run}")

    all_results = {}

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*60}")

        # HPO on split 0 (skip for models with fixed params)
        train_mask_0 = data.train_mask[:, 0].to(device)
        val_mask_0   = data.val_mask[:,   0].to(device)

        best_params, best_val_f1 = run_hpo(
            model_name, x, y, edge_index,
            train_mask_0, val_mask_0,
            num_classes, args.num_trials, device
        )

        # Evaluate best params on all 10 splits
        print(f"\n  Evaluating best params across {NUM_SPLITS} splits...")
        split_results = []
        for split_idx in range(NUM_SPLITS):
            train_mask = data.train_mask[:, split_idx].to(device)
            val_mask   = data.val_mask[:,   split_idx].to(device)
            test_mask  = data.test_mask[:,  split_idx].to(device)

            result = run_one_split(
                model_name, best_params,
                x, y, edge_index,
                train_mask, val_mask, test_mask,
                num_classes, device, split_idx
            )
            split_results.append(result)

        all_results[model_name] = {
            "best_params":  best_params,
            "best_val_f1":  best_val_f1,
            "splits":       split_results,
        }

    # Save everything
    print("\nSaving results...")
    df_summary, df_class = save_results(all_results, num_classes, args.out_dir)
    print_summary_table(df_summary)
    save_macro_f1_plot(df_summary, args.out_dir)
    save_per_class_f1_plot(df_class, args.out_dir)

    # Also dump raw results as JSON (handy for later analysis)
    json_path = os.path.join(args.out_dir, "tables", "all_results.json")
    json_safe = {}
    for model_name, res in all_results.items():
        json_safe[model_name] = {
            "best_params": res["best_params"],
            "best_val_f1": res["best_val_f1"],
            "splits": [
                {k: v.tolist() if hasattr(v, "tolist") else v
                 for k, v in s.items() if k != "report"}
                for s in res["splits"]
            ],
        }
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"  [saved] {json_path}")

    print(f"\nDone. All outputs in: {args.out_dir}")
    print(f"Run finished: {datetime.datetime.now().isoformat()}")
    sys.stdout = sys.stdout._stream
    _log_file.close()


if __name__ == "__main__":
    main()
