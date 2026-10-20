# Feature preparation, loss and training for the multi-asset Panel GAT.

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn import functional as F
from torch_geometric.data import Data

from crypto_gnn.data.preparation.features import (
    FEATURE_SET_NAMES,
    build_causal_features,
    fit_feature_standardizer,
    standardize_features,
)
from crypto_gnn.graph.panel import EDGE_RELATIONAL
from crypto_gnn.graph.temporal import EDGE_TEMPORAL
from crypto_gnn.models import PanelGAT
from crypto_gnn.training.common import compute_class_weights, set_reproducible_seed


# Rebuild split masks on panel nodes because rows missing for one asset do not
# become nodes. Purging removes labels whose future price crosses a boundary.
def build_panel_split_masks(graph, split, forecast_horizon_hours=1):
    horizon = forecast_horizon_hours * 3_600_000_000_000
    label_times = graph["timestamps"].asi8 + horizon
    valid = graph["target_mask"]

    train_mask = (
        (graph["timestamps"] >= split["train_start"])
        & (graph["timestamps"] < split["validation_start"])
        & (label_times < split["validation_start"].value)
        & valid
    )
    validation_mask = (
        (graph["timestamps"] >= split["validation_start"])
        & (graph["timestamps"] < split["test_start"])
        & (label_times < split["test_start"].value)
        & valid
    )
    test_mask = (
        (graph["timestamps"] >= split["test_start"])
        & (graph["timestamps"] < split["test_end"])
        & (label_times < split["test_end"].value)
        & valid
    )
    return (
        np.asarray(train_mask, dtype=bool),
        np.asarray(validation_mask, dtype=bool),
        np.asarray(test_mask, dtype=bool),
    )


# Transform and standardize each cryptocurrency independently. Training nodes from
# that asset alone estimate its mean and scale, so validation and test stay unseen.
def build_standardized_panel_features(
    datasets,
    graph,
    split,
    feature_set="ohlc",
    forecast_horizon_hours=1,
):
    train_mask, _, _ = build_panel_split_masks(
        graph,
        split,
        forecast_horizon_hours,
    )
    values = np.empty(
        (graph["number_of_nodes"], len(FEATURE_SET_NAMES[feature_set])),
        dtype=np.float32,
    )
    metadata = {}

    for symbol_index, symbol in enumerate(graph["symbols"]):
        node_mask = graph["node_symbol_indices"] == symbol_index
        node_ids = np.flatnonzero(node_mask)
        timestamps = graph["timestamps"][node_mask]
        observed = datasets[symbol].loc[timestamps]
        raw_features = build_causal_features(observed, feature_set)
        local_training_mask = train_mask[node_mask]

        standardizer = fit_feature_standardizer(
            raw_features,
            local_training_mask,
        )
        values[node_ids] = standardize_features(raw_features, standardizer)
        metadata[symbol] = {
            "mean": standardizer["mean"].tolist(),
            "scale": standardizer["scale"].tolist(),
            "fit_nodes": int(local_training_mask.sum()),
            "ddof": 0,
        }

    return {
        "values": values,
        "feature_names": FEATURE_SET_NAMES[feature_set],
        "standardization_by_symbol": metadata,
    }


# Keep two hours of past context for the two GAT layers, then convert global panel
# node identifiers to the contiguous local identifiers required by PyG.
def prepare_panel_fold_data(
    graph,
    node_features,
    split,
    message_passing_layers=2,
    forecast_horizon_hours=1,
):
    masks = build_panel_split_masks(
        graph,
        split,
        forecast_horizon_hours,
    )
    context_start = split["train_start"] - pd.DateOffset(
        hours=message_passing_layers
    )
    selected = (
        (graph["timestamps"] >= context_start)
        & (graph["timestamps"] < split["test_end"])
    )
    global_nodes = np.flatnonzero(selected)

    global_to_local = np.full(graph["number_of_nodes"], -1, dtype=np.int64)
    global_to_local[global_nodes] = np.arange(len(global_nodes), dtype=np.int64)
    sources, destinations = graph["edge_index"]
    edge_mask = selected[sources] & selected[destinations]
    local_edges = global_to_local[graph["edge_index"][:, edge_mask]]
    local_edge_type = graph["edge_type"][edge_mask]

    data = Data(
        x=torch.from_numpy(
            np.asarray(node_features[global_nodes], dtype=np.float32)
        ),
        edge_index=torch.from_numpy(local_edges.astype(np.int64, copy=False)),
        edge_type=torch.from_numpy(local_edge_type.astype(np.int64, copy=False)),
        y=torch.from_numpy(
            graph["targets"][global_nodes].astype(np.int64, copy=False)
        ),
        node_asset_indices=torch.from_numpy(
            graph["node_symbol_indices"][global_nodes].astype(
                np.int64,
                copy=False,
            )
        ),
        train_mask=torch.from_numpy(masks[0][global_nodes]),
        validation_mask=torch.from_numpy(masks[1][global_nodes]),
        test_mask=torch.from_numpy(masks[2][global_nodes]),
        global_node_indices=torch.from_numpy(global_nodes),
    )
    data.symbols = graph["symbols"]
    return data


# Calculate inverse-frequency class weights separately for each asset. This avoids
# letting one cryptocurrency's class balance determine every other asset's loss.
def compute_asset_class_weights(
    targets,
    mask,
    node_asset_indices,
    number_of_assets,
    classes=3,
):
    weights = []
    for asset in range(number_of_assets):
        asset_mask = mask & (node_asset_indices == asset)
        weights.append(compute_class_weights(targets, asset_mask, classes))
    return torch.stack(weights)


# Average one weighted cross-entropy per asset. Equal asset weighting prevents a
# cryptocurrency with more valid nodes from dominating the shared backbone.
def mean_asset_cross_entropy(
    logits,
    targets,
    mask,
    node_asset_indices,
    class_weights,
):
    losses = []
    for asset in range(len(class_weights)):
        asset_mask = mask & (node_asset_indices == asset)
        loss = F.cross_entropy(
            logits[asset_mask],
            targets[asset_mask],
            weight=class_weights[asset],
        )
        losses.append(loss)
    return torch.stack(losses).mean()


# Average the three-class macro F1 values of all assets. This is the selection and
# early-stopping metric used by the relational experiment.
def mean_asset_macro_f1(
    logits,
    targets,
    mask,
    node_asset_indices,
    number_of_assets,
):
    predictions = logits.argmax(dim=1)
    scores = []
    for asset in range(number_of_assets):
        asset_mask = mask & (node_asset_indices == asset)
        actual = targets[asset_mask].detach().cpu().numpy()
        predicted = predictions[asset_mask].detach().cpu().numpy()
        score = f1_score(
            actual,
            predicted,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
        scores.append(score)
    return float(np.mean(scores))


# Train the shared Panel GAT and select its checkpoint using validation data only.
def train_panel_gat(data, config, symbols):
    set_reproducible_seed(config["seed"])
    torch.use_deterministic_algorithms(
        config.get("deterministic_algorithms", True)
    )
    device = torch.device(config["device"])
    data = data.to(device)
    number_of_assets = len(symbols)

    model = PanelGAT(
        input_channels=data.num_node_features,
        hidden_channels=config["hidden_channels"],
        attention_heads=config["attention_heads"],
        number_of_assets=number_of_assets,
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    class_weights = compute_asset_class_weights(
        data.y,
        data.train_mask,
        data.node_asset_indices,
        number_of_assets,
    ).to(device)

    best_state = None
    best_validation = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["maximum_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.node_asset_indices)
        training_loss = mean_asset_cross_entropy(
            logits,
            data.y,
            data.train_mask,
            data.node_asset_indices,
            class_weights,
        )
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Validation is evaluated after every update, while test nodes remain
        # untouched until the best checkpoint has been restored.
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.node_asset_indices)
            validation_loss = mean_asset_cross_entropy(
                logits,
                data.y,
                data.validation_mask,
                data.node_asset_indices,
                class_weights,
            )
            validation_f1 = mean_asset_macro_f1(
                logits,
                data.y,
                data.validation_mask,
                data.node_asset_indices,
                number_of_assets,
            )

        history.append(
            {
                "epoch": epoch,
                "training_loss": float(training_loss.item()),
                "validation_loss": float(validation_loss.item()),
                "validation_mean_asset_macro_f1": validation_f1,
            }
        )
        if validation_f1 > best_validation + 1e-6:
            best_validation = validation_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.node_asset_indices)
        predictions = logits.argmax(dim=1)

    test_mean_asset_f1 = mean_asset_macro_f1(
        logits,
        data.y,
        data.test_mask,
        data.node_asset_indices,
        number_of_assets,
    )
    pooled_actual = data.y[data.test_mask].detach().cpu().numpy()
    pooled_predictions = predictions[data.test_mask].detach().cpu().numpy()
    pooled_macro_f1 = float(
        f1_score(
            pooled_actual,
            pooled_predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )

    by_symbol = {}
    for asset, symbol in enumerate(symbols):
        asset_test = data.test_mask & (data.node_asset_indices == asset)
        actual = data.y[asset_test].detach().cpu().numpy()
        predicted = predictions[asset_test].detach().cpu().numpy()
        by_symbol[symbol] = {
            "test_nodes": int(asset_test.sum().item()),
            "macro_f1": float(
                f1_score(
                    actual,
                    predicted,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
            "accuracy": float(np.mean(actual == predicted)),
            "confusion_matrix": confusion_matrix(
                actual,
                predicted,
                labels=[0, 1, 2],
            ).tolist(),
        }

    result = {
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "validation_mean_asset_macro_f1": best_validation,
        "test_mean_asset_macro_f1": test_mean_asset_f1,
        "test_pooled_macro_f1": pooled_macro_f1,
        "test_by_symbol": by_symbol,
        "history": history,
    }
    return model, result
