"""
Training loop and evaluation shared by all five models.
All models accept forward(x, edge_index=None), so one loop handles everything.
"""

import copy
import torch
from sklearn.metrics import f1_score


def get_class_weights(y, train_mask, num_classes, device):
    """
    Inverse-frequency weights computed from training nodes only.
    Rescaled so the mean weight equals 1 (keeps loss magnitudes stable).
    bincount runs on CPU for MPS safety — it's a tiny op anyway.
    """
    y_train = y[train_mask].cpu()
    counts = torch.bincount(y_train, minlength=num_classes).float()
    weights = 1.0 / counts.clamp(min=1.0)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def train_one_epoch(model, x, y, edge_index, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, x, y, edge_index, mask, criterion):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    loss = criterion(out[mask], y[mask]).item()
    preds = out[mask].argmax(dim=1).cpu().numpy()
    labels = y[mask].cpu().numpy()
    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return loss, acc, macro_f1, preds, labels


def train_model(model, x, y, edge_index, train_mask, val_mask,
                optimizer, criterion, max_epochs=1000, patience=100):
    """
    Train with early stopping on val macro-F1.
    Saves the best model state and restores it before returning.
    Returns: (best_val_macro_f1, number_of_epochs_run)
    """
    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    # Reduce LR when val loss plateaus — same idea as geom-gcn
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    for epoch in range(max_epochs):
        train_one_epoch(model, x, y, edge_index, train_mask, optimizer, criterion)
        val_loss, _, val_f1, _, _ = evaluate(model, x, y, edge_index, val_mask, criterion)

        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return best_val_f1, epoch + 1
