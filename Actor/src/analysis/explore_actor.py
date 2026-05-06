"""
Milestone 1 – Exploratory analysis of the Actor dataset.

Run:
    python src/analysis/explore_actor.py --data_root data/actor \
        --out_dir reports/milestone1 --seed 42
"""

import argparse
import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch_geometric.utils import (
    is_undirected,
    contains_self_loops,
    to_networkx,
)
import networkx as nx

# Make sure the project root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.load_actor import load_actor
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


def _save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [saved] {path}")


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


# ---------------------------------------------------------------------------
# A. Dataset loading and sanity checks
# ---------------------------------------------------------------------------

def section_a(data, dataset, out_dir: str) -> dict:
    print("\n=== A. Dataset overview ===")

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    num_features = data.num_node_features
    num_classes = dataset.num_classes

    undirected = is_undirected(data.edge_index, num_nodes=num_nodes)
    has_self_loops = contains_self_loops(data.edge_index)

    # Number of predefined splits (train_mask may have shape [N] or [N, K])
    if data.train_mask.dim() == 1:
        num_splits = 1
    else:
        num_splits = data.train_mask.shape[1]

    # Feature sparsity
    x = data.x  # shape [N, F]
    assert x.shape == (num_nodes, num_features), "x shape mismatch"
    frac_zeros = (x == 0).float().mean().item()
    active_per_node = (x != 0).float().sum(dim=1)  # [N]
    avg_active = active_per_node.mean().item()
    min_active = active_per_node.min().item()
    max_active = active_per_node.max().item()
    is_binary = bool(x.unique().numel() <= 2)

    overview = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": num_features,
        "num_classes": num_classes,
        "is_undirected": undirected,
        "has_self_loops": has_self_loops,
        "x_shape": list(x.shape),
        "edge_index_shape": list(data.edge_index.shape),
        "y_shape": list(data.y.shape),
        "train_mask_shape": list(data.train_mask.shape),
        "val_mask_shape": list(data.val_mask.shape),
        "test_mask_shape": list(data.test_mask.shape),
        "num_splits": num_splits,
        "features_binary": is_binary,
        "feature_sparsity_frac_zeros": round(frac_zeros, 4),
        "avg_active_features_per_node": round(avg_active, 2),
        "min_active_features": int(min_active),
        "max_active_features": int(max_active),
    }

    for k, v in overview.items():
        print(f"  {k}: {v}")

    df = pd.DataFrame([overview])
    _save_csv(df, os.path.join(out_dir, "tables", "overview.csv"))
    return overview


# ---------------------------------------------------------------------------
# B. Label and split analysis
# ---------------------------------------------------------------------------

def section_b(data, dataset, out_dir: str) -> dict:
    print("\n=== B. Label and split analysis ===")

    y = data.y.numpy()
    num_classes = dataset.num_classes
    num_nodes = data.num_nodes

    # Class counts (always use split 0 if multi-split)
    class_counts = np.bincount(y, minlength=num_classes)
    class_props = class_counts / num_nodes

    df_class = pd.DataFrame({
        "class": list(range(num_classes)),
        "count": class_counts,
        "proportion": np.round(class_props, 4),
    })
    print(df_class.to_string(index=False))
    _save_csv(df_class, os.path.join(out_dir, "tables", "class_distribution.csv"))

    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df_class["class"], df_class["count"], color="steelblue", edgecolor="black")
    ax.set_xlabel("Class")
    ax.set_ylabel("Node count")
    ax.set_title("Actor – Class Distribution")
    ax.set_xticks(df_class["class"])
    _save_fig(fig, os.path.join(out_dir, "figures", "class_distribution.png"))

    # Split sizes – handle both single and multi-split masks
    def _split_count(mask):
        if mask.dim() == 1:
            return int(mask.sum().item())
        else:
            # return counts per split
            return mask.sum(dim=0).tolist()

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    if train_mask.dim() == 1:
        split_rows = [{"split_idx": 0,
                       "train": int(train_mask.sum()),
                       "val": int(val_mask.sum()),
                       "test": int(test_mask.sum())}]
    else:
        num_splits = train_mask.shape[1]
        split_rows = []
        for i in range(num_splits):
            split_rows.append({
                "split_idx": i,
                "train": int(train_mask[:, i].sum()),
                "val": int(val_mask[:, i].sum()),
                "test": int(test_mask[:, i].sum()),
            })

    df_splits = pd.DataFrame(split_rows)
    print(df_splits.to_string(index=False))
    _save_csv(df_splits, os.path.join(out_dir, "tables", "split_sizes.csv"))

    # Split bar chart (first split only)
    row0 = split_rows[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Train", "Val", "Test"],
           [row0["train"], row0["val"], row0["test"]],
           color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black")
    ax.set_ylabel("Node count")
    ax.set_title("Actor – Split Sizes (split 0)")
    _save_fig(fig, os.path.join(out_dir, "figures", "split_sizes.png"))

    return {"class_distribution": df_class, "splits": df_splits}


# ---------------------------------------------------------------------------
# C. Graph topology
# ---------------------------------------------------------------------------

def section_c(data, out_dir: str) -> dict:
    print("\n=== C. Graph topology ===")

    num_nodes = data.num_nodes
    num_edges = data.num_edges

    # Degree from edge_index (undirected: count each endpoint)
    row, col = data.edge_index
    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.long))
    # For undirected graphs edges appear twice so degree is already correct.

    deg_np = degree.numpy()
    min_deg = int(deg_np.min())
    mean_deg = float(deg_np.mean())
    median_deg = float(np.median(deg_np))
    max_deg = int(deg_np.max())

    # Graph density: edges / (N*(N-1)) for undirected
    density = num_edges / (num_nodes * (num_nodes - 1))

    # Connected components via NetworkX
    print("  Building NetworkX graph for component analysis…")
    G = to_networkx(data, to_undirected=True)
    components = list(nx.connected_components(G))
    num_components = len(components)
    largest_cc_size = max(len(c) for c in components)
    isolated_nodes = sum(1 for c in components if len(c) == 1)

    topo = {
        "min_degree": min_deg,
        "mean_degree": round(mean_deg, 2),
        "median_degree": round(median_deg, 2),
        "max_degree": max_deg,
        "graph_density": f"{density:.6f}",
        "num_connected_components": num_components,
        "largest_cc_size": largest_cc_size,
        "isolated_nodes": isolated_nodes,
    }
    for k, v in topo.items():
        print(f"  {k}: {v}")

    df_topo = pd.DataFrame([topo])
    _save_csv(df_topo, os.path.join(out_dir, "tables", "topology.csv"))

    # Degree histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = min(50, max_deg - min_deg + 1)
    ax.hist(deg_np, bins=bins, color="steelblue", edgecolor="black", log=True)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Actor – Degree Distribution")
    ax.axvline(mean_deg, color="red", linestyle="--", label=f"mean={mean_deg:.1f}")
    ax.legend()
    _save_fig(fig, os.path.join(out_dir, "figures", "degree_distribution.png"))

    return topo


# ---------------------------------------------------------------------------
# D. Homophily analysis
# ---------------------------------------------------------------------------

def section_d(data, out_dir: str) -> dict:
    print("\n=== D. Homophily analysis ===")

    row, col = data.edge_index
    y = data.y

    # Edge homophily: fraction of edges where src and dst share label
    same_label = (y[row] == y[col]).float()
    edge_homophily = same_label.mean().item()

    # Per-node neighborhood purity
    num_nodes = data.num_nodes
    same_count = torch.zeros(num_nodes)
    total_count = torch.zeros(num_nodes)
    same_count.scatter_add_(0, row, same_label)
    total_count.scatter_add_(0, row, torch.ones(row.size(0)))

    # Avoid division by zero for isolated nodes
    has_neighbors = total_count > 0
    node_purity = torch.zeros(num_nodes)
    node_purity[has_neighbors] = same_count[has_neighbors] / total_count[has_neighbors]

    purity_np = node_purity[has_neighbors].numpy()
    mean_purity = float(purity_np.mean())
    median_purity = float(np.median(purity_np))

    homophily = {
        "edge_homophily": round(edge_homophily, 4),
        "mean_node_purity": round(mean_purity, 4),
        "median_node_purity": round(median_purity, 4),
    }
    for k, v in homophily.items():
        print(f"  {k}: {v}")

    df_h = pd.DataFrame([homophily])
    _save_csv(df_h, os.path.join(out_dir, "tables", "homophily.csv"))

    # Node purity histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(purity_np, bins=30, color="coral", edgecolor="black")
    ax.set_xlabel("Neighborhood purity")
    ax.set_ylabel("Node count")
    ax.set_title("Actor – Per-Node Neighborhood Purity")
    ax.axvline(mean_purity, color="blue", linestyle="--", label=f"mean={mean_purity:.2f}")
    ax.legend()
    _save_fig(fig, os.path.join(out_dir, "figures", "neighborhood_purity.png"))

    return homophily


# ---------------------------------------------------------------------------
# F. Markdown summary
# ---------------------------------------------------------------------------

def section_f(overview: dict, homophily: dict, topo: dict, out_dir: str) -> None:
    print("\n=== F. Generating Markdown summary ===")

    edge_h = homophily["edge_homophily"]
    homo_verdict = "heterophilic" if edge_h < 0.5 else "homophilic"

    md = textwrap.dedent(f"""\
    # Actor Dataset – Milestone 1 Summary

    ## 1. What the Actor Dataset Represents
    The Actor dataset is a web-scraped graph where each **node** represents an actor
    and each **edge** connects two actors who appear together on the same Wikipedia page.
    Node features are bag-of-words keyword indicators derived from the actors' Wikipedia pages.
    There are **{overview['num_nodes']}** nodes, **{overview['num_edges']}** edges, and
    **{overview['num_features']}** features per node.

    ## 2. Why the Problem is Node Classification
    Each actor is assigned one of **{overview['num_classes']}** topic categories.
    The goal is to predict the category of every actor using the graph structure and
    node features — a standard transductive node classification task.

    ## 3. Why Node Features Are Necessary
    Without node features, each actor is indistinguishable from another.
    The keyword indicators encode what Wikipedia pages the actor appears on, providing
    content-based signals that a model can use to separate classes.
    Features are {'binary' if overview['features_binary'] else 'real-valued'} and sparse
    (fraction of zeros: **{overview['feature_sparsity_frac_zeros']:.1%}**;
    average active features per node: **{overview['avg_active_features_per_node']}**).

    ## 4. Why Graph Edges Are Necessary
    Co-occurrence edges carry relational context: if two actors frequently appear on the
    same Wikipedia pages they are likely in related topic areas.
    A purely feature-based classifier ignores this relational signal.
    However, as discussed below, the graph is heterophilic, so message passing must be
    used carefully.

    ## 5. Homophily Analysis
    | Metric | Value |
    |---|---|
    | Edge homophily ratio | {edge_h:.4f} |
    | Mean node neighborhood purity | {homophily['mean_node_purity']:.4f} |
    | Median node neighborhood purity | {homophily['median_node_purity']:.4f} |

    An edge homophily ratio of **{edge_h:.4f}** (well below 0.5) indicates the graph is
    **{homo_verdict}**: connected actors tend to belong to *different* classes.
    This means standard GNN message-passing (which aggregates neighbor labels) may
    hurt rather than help — a key challenge for Milestone 2.

    ## 6. Challenges for GNNs
    - **Heterophily**: Low edge homophily means most neighbors carry noise, not signal.
      Standard GCN/GraphSAGE may under-perform MLP baselines.
    - **Sparse features**: Only {100*(1-overview['feature_sparsity_frac_zeros']):.1f}% of
      feature entries are non-zero; models must handle high-dimensional sparse input.
    - **Class imbalance**: Some classes have substantially fewer nodes (see class
      distribution table), which can bias learned representations.
    - **Graph topology**: The graph has **{topo['num_connected_components']}** connected
      component(s); the largest contains **{topo['largest_cc_size']}** nodes.
      **{topo['isolated_nodes']}** isolated node(s) have no neighbors, so they rely
      entirely on their own features.
    - **Mean degree {topo['mean_degree']}**: Relatively sparse connectivity limits
      how much structural information can propagate.

    ## Files Generated
    | Path | Description |
    |---|---|
    | `reports/milestone1/tables/overview.csv` | Dataset-level statistics |
    | `reports/milestone1/tables/class_distribution.csv` | Per-class counts and proportions |
    | `reports/milestone1/tables/split_sizes.csv` | Train/val/test node counts |
    | `reports/milestone1/tables/topology.csv` | Degree and connectivity statistics |
    | `reports/milestone1/tables/homophily.csv` | Edge homophily and node purity |
    | `reports/milestone1/figures/class_distribution.png` | Class distribution bar chart |
    | `reports/milestone1/figures/split_sizes.png` | Split sizes bar chart |
    | `reports/milestone1/figures/degree_distribution.png` | Degree histogram |
    | `reports/milestone1/figures/neighborhood_purity.png` | Node purity histogram |
    """)

    out_path = os.path.join(out_dir, "actor_dataset_summary.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Milestone 1 – Actor dataset exploration")
    parser.add_argument("--data_root", type=str, default="data/actor",
                        help="Directory to store/load the Actor dataset")
    parser.add_argument("--out_dir", type=str, default="reports/milestone1",
                        help="Directory for output reports and figures")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)

    print("Loading Actor dataset…")
    dataset = load_actor(root=args.data_root)
    data = dataset[0]
    print(f"  Graph: {data}")

    overview = section_a(data, dataset, args.out_dir)
    section_b(data, dataset, args.out_dir)
    topo = section_c(data, args.out_dir)
    homophily = section_d(data, args.out_dir)
    section_f(overview, homophily, topo, args.out_dir)

    print("\nDone. All outputs saved to:", args.out_dir)


if __name__ == "__main__":
    main()
