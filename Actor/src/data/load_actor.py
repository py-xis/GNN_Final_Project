"""Load the Actor dataset using PyTorch Geometric."""

import os
from torch_geometric.datasets import Actor


def load_actor(root: str = "data/actor") -> Actor:
    """
    Download (if needed) and return the Actor dataset.

    Args:
        root: Directory where the dataset will be stored.

    Returns:
        The Actor dataset object.
    """
    dataset = Actor(root=root)
    assert len(dataset) == 1, "Actor dataset should contain exactly one graph."
    data = dataset[0]

    # Basic sanity checks
    assert data.x is not None, "Node features (x) are missing."
    assert data.edge_index is not None, "Edge index is missing."
    assert data.y is not None, "Node labels (y) are missing."
    assert data.train_mask is not None, "Train mask is missing."
    assert data.val_mask is not None, "Val mask is missing."
    assert data.test_mask is not None, "Test mask is missing."

    return dataset


if __name__ == "__main__":
    dataset = load_actor()
    data = dataset[0]
    print(f"Dataset loaded: {dataset}")
    print(f"Number of nodes   : {data.num_nodes}")
    print(f"Number of edges   : {data.num_edges}")
    print(f"Number of features: {data.num_node_features}")
    print(f"Number of classes : {dataset.num_classes}")
