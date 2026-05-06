"""
Hyper-parameter optimisation using Optuna (TPE sampler — Bayesian optimisation).
Runs num_trials trials on split 0 and returns the best parameter config
measured by validation macro-F1.
"""

import torch
import torch.nn as nn
import optuna

from src.models.models import LogisticRegression, MLP, GCN, GraphSAGE, GAT, APPNPNet
from src.train import get_class_weights, train_model, evaluate
from src.utils.seed import set_seed

# Silence optuna's per-trial logs — we print our own summary
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_model(model_name, params, in_features, num_classes, device):
    """Instantiate a model from a param dict."""
    if model_name == "lr_model":
        model = LogisticRegression(in_features, num_classes, dropout=params["dropout"])
    elif model_name == "mlp":
        model = MLP(in_features, num_classes,
                    params["hidden_dim"], params["num_layers"], params["dropout"])
    elif model_name == "gcn":
        model = GCN(in_features, num_classes,
                    params["hidden_dim"], params["num_layers"], params["dropout"])
    elif model_name == "sage":
        model = GraphSAGE(in_features, num_classes,
                          params["hidden_dim"], params["num_layers"],
                          params["dropout"], params["aggr"])
    elif model_name == "gat":
        model = GAT(in_features, num_classes,
                    params["hidden_dim"], params["num_layers"],
                    params["heads"], params["dropout"])
    elif model_name == "appnp":
        model = APPNPNet(in_features, num_classes,
                         params["hidden_dim"], params["mlp_layers"],
                         params["dropout"], params["K"], params["alpha"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def _suggest_params(trial, model_name):
    """Let Optuna suggest a parameter config for one trial."""
    if model_name == "lr_model":
        return {
            "lr":           trial.suggest_float("lr",          1e-3, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "dropout":      trial.suggest_float("dropout",      0.0,  0.5,  step=0.1),
        }
    elif model_name == "mlp":
        return {
            "hidden_dim":   trial.suggest_categorical("hidden_dim",   [64, 128, 256]),
            "num_layers":   trial.suggest_int("num_layers", 2, 3),
            "dropout":      trial.suggest_float("dropout",  0.0, 0.5, step=0.1),
            "lr":           trial.suggest_float("lr",        1e-3, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }
    elif model_name == "gcn":
        return {
            "hidden_dim":   trial.suggest_categorical("hidden_dim",   [64, 128, 256]),
            "num_layers":   trial.suggest_int("num_layers", 2, 3),
            "dropout":      trial.suggest_float("dropout",  0.0, 0.5, step=0.1),
            "lr":           trial.suggest_float("lr",        1e-3, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }
    elif model_name == "sage":
        return {
            "hidden_dim":   trial.suggest_categorical("hidden_dim",   [64, 128, 256]),
            "num_layers":   trial.suggest_int("num_layers", 2, 3),
            "aggr":         trial.suggest_categorical("aggr", ["mean", "max"]),
            "dropout":      trial.suggest_float("dropout",  0.0, 0.5, step=0.1),
            "lr":           trial.suggest_float("lr",        1e-3, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }
    elif model_name == "gat":
        return {
            "hidden_dim":   trial.suggest_categorical("hidden_dim", [8, 16, 32]),
            "num_layers":   trial.suggest_int("num_layers", 2, 3),
            "heads":        trial.suggest_categorical("heads", [4, 8]),
            "dropout":      trial.suggest_float("dropout",  0.0, 0.6, step=0.1),
            "lr":           trial.suggest_float("lr",        1e-3, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }
    elif model_name == "appnp":
        return {
            "hidden_dim":   trial.suggest_categorical("hidden_dim",   [64, 128, 256]),
            "mlp_layers":   trial.suggest_int("mlp_layers", 1, 2),
            "dropout":      trial.suggest_float("dropout",  0.0, 0.5, step=0.1),
            "lr":           trial.suggest_float("lr",        1e-3, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "K":            trial.suggest_categorical("K",     [5, 10, 20]),
            "alpha":        trial.suggest_categorical("alpha", [0.1, 0.2, 0.5]),
        }
    else:
        raise ValueError(f"No search space defined for: {model_name}")


def run_hpo(model_name, x, y, edge_index, train_mask, val_mask,
            num_classes, num_trials, device):
    """
    Bayesian HPO with Optuna's TPE sampler.
    Returns (best_params dict, best val macro-F1).
    """
    print(f"\n  Running Optuna HPO: {num_trials} trials for [{model_name}]")

    class_weights = get_class_weights(y, train_mask, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    def objective(trial):
        params = _suggest_params(trial, model_name)
        set_seed(42 + trial.number)

        model = build_model(model_name, params, x.shape[1], num_classes, device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        val_f1, epochs = train_model(
            model, x, y, edge_index, train_mask, val_mask,
            optimizer, criterion, max_epochs=1000, patience=100
        )

        print(f"    trial {trial.number+1:2d}/{num_trials} | "
              f"val macro-F1: {val_f1:.4f} | epochs: {epochs:4d} | params: {params}")
        return val_f1

    # Use fewer startup trials so TPE's Bayesian logic kicks in earlier,
    # which matters when num_trials is small (e.g. 25).
    n_startup = max(5, num_trials // 5)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=n_startup),
    )
    study.optimize(objective, n_trials=num_trials)

    best_params = study.best_params
    best_val_f1 = study.best_value

    print(f"  => Best val macro-F1: {best_val_f1:.4f}")
    print(f"  => Best params: {best_params}")
    return best_params, best_val_f1
