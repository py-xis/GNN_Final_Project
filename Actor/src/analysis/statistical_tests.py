"""
Paired statistical tests on the 10-split macro-F1 results.

Reads all_results.json, runs paired Wilcoxon signed-rank tests between key
model pairs, and prints a formatted table.  Also prints the minority-class
macro-F1 (mean F1 over classes 0-3, excluding majority class 4) per model.

Run from project root:
    python src/analysis/statistical_tests.py \
        --results_json reports/milestone2/tables/all_results.json \
        --per_class_csv reports/milestone2/tables/per_class_f1.csv
"""

import argparse
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_split_f1(results_json: str):
    """Return dict: model -> np.array of 10 test macro-F1 values."""
    with open(results_json) as f:
        data = json.load(f)
    out = {}
    for model, res in data.items():
        out[model] = np.array([s["test_macro_f1"] for s in res["splits"]])
    return out


def paired_wilcoxon_table(split_f1: dict):
    """All-pairs paired Wilcoxon; return DataFrame."""
    models = list(split_f1.keys())
    rows = []
    for a, b in itertools.combinations(models, 2):
        fa, fb = split_f1[a], split_f1[b]
        diff = fa - fb
        # Wilcoxon requires non-zero differences
        if np.all(diff == 0):
            stat, pval = np.nan, 1.0
        else:
            stat, pval = wilcoxon(fa, fb, alternative="two-sided", zero_method="wilcox")
        rows.append({
            "model_A": a, "model_B": b,
            "mean_A":  round(fa.mean(), 4), "std_A": round(fa.std(), 4),
            "mean_B":  round(fb.mean(), 4), "std_B": round(fb.std(), 4),
            "delta_A_minus_B": round((fa - fb).mean(), 4),
            "p_value": round(pval, 4),
            "significant_p05": pval < 0.05,
        })
    return pd.DataFrame(rows)


def minority_macro_f1(per_class_csv: str, minority_classes=(0, 1, 2, 3)):
    """Mean F1 over minority classes per model."""
    df = pd.read_csv(per_class_csv)
    minority = df[df["class"].isin(minority_classes)]
    out = minority.groupby("model")["f1_mean"].mean().rename("minority_macro_f1").round(4)
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json",  default="reports/milestone2/tables/all_results.json")
    parser.add_argument("--per_class_csv", default="reports/milestone2/tables/per_class_f1.csv")
    args = parser.parse_args()

    split_f1 = load_split_f1(args.results_json)

    print("\n" + "=" * 70)
    print("PER-MODEL MACRO-F1 ACROSS 10 SPLITS")
    print("=" * 70)
    for model, vals in split_f1.items():
        print(f"  {model:<10}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")

    print("\n" + "=" * 70)
    print("PAIRED WILCOXON SIGNED-RANK TEST (two-sided, all pairs)")
    print("=" * 70)
    wdf = paired_wilcoxon_table(split_f1)
    # Sort by |delta| descending
    wdf = wdf.reindex(wdf["delta_A_minus_B"].abs().sort_values(ascending=False).index)
    for _, row in wdf.iterrows():
        sig = "**" if row["significant_p05"] else "  "
        print(f"  {sig} {row['model_A']:<10} vs {row['model_B']:<10} | "
              f"delta={row['delta_A_minus_B']:+.4f} | "
              f"A={row['mean_A']:.4f}±{row['std_A']:.4f}  "
              f"B={row['mean_B']:.4f}±{row['std_B']:.4f} | "
              f"p={row['p_value']:.4f} {'(sig)' if row['significant_p05'] else ''}")

    # Highlight the key pairs
    key_pairs = [
        ("mlp", "sage"), ("mlp", "appnp"), ("sage", "appnp"),
        ("mlp", "gcn"),  ("mlp", "gat"),   ("gcn", "gat"),
        ("mlp", "lr_model"),
    ]
    print("\n" + "=" * 70)
    print("KEY PAIRS (for report)")
    print("=" * 70)
    for a, b in key_pairs:
        row = wdf[(wdf["model_A"] == a) & (wdf["model_B"] == b)]
        if row.empty:
            row = wdf[(wdf["model_A"] == b) & (wdf["model_B"] == a)]
            if row.empty:
                continue
            row = row.copy()
            row["delta_A_minus_B"] = -row["delta_A_minus_B"]
            a, b = b, a
        r = row.iloc[0]
        sig = "SIGNIFICANT" if r["significant_p05"] else "not significant"
        print(f"  {a} vs {b}: delta={r['delta_A_minus_B']:+.4f}  p={r['p_value']:.4f}  [{sig}]")

    print("\n" + "=" * 70)
    print("MINORITY-CLASS MACRO-F1 (classes 0-3, excl. majority class 4)")
    print("=" * 70)
    mdf = minority_macro_f1(args.per_class_csv)
    mdf = mdf.sort_values("minority_macro_f1", ascending=False)
    for _, row in mdf.iterrows():
        print(f"  {row['model']:<10}  minority_macro_f1 = {row['minority_macro_f1']:.4f}")

    # Save Wilcoxon table
    out_path = "reports/milestone2/tables/wilcoxon_tests.csv"
    wdf.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
