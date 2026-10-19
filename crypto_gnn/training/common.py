# Shared preparation, training and evaluation for the independent neural models.

import copy
import random

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.nn import functional as F
from torch_geometric.data import Data

from crypto_gnn.graph.temporal import EDGE_TEMPORAL
from crypto_gnn.models import GATBaseline, MLPBaseline


# Reset every random generator before a fit so repeated runs with the same seed
# start from the same weights and use the same dropout sequence.
def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Give rare classes a larger loss weight. Counts come only from the training mask,
# so validation and test class frequencies cannot influence optimization.
def compute_class_weights(targets, mask, classes=3):
    selected = targets[mask]
    counts = torch.bincount(selected, minlength=classes).to(torch.float32)
    weights = counts.sum() / (classes * counts)
    return weights / weights.mean()


# Convert a global temporal split to a smaller PyG graph. A limited amount of past
# context is retained because the first training nodes still receive GAT messages.
def prepare_fold_data(
    graph,
    node_features,
    split,
    edge_mode="temporal_visibility",
    message_passing_layers=2,
):
    source_positions = graph["source_positions"]
    graph_train_mask = split["train_mask"][source_positions]
    graph_validation_mask = split["validation_mask"][source_positions]
    graph_test_mask = split["test_mask"][source_positions]

    train_positions = np.flatnonzero(graph_train_mask)
    test_positions = np.flatnonzero(graph_test_mask)
    sources = graph["edge_index"][0]
    targets = graph["edge_index"][1]

    # With two GAT layers, information can cross two edges. The largest edge jump
    # gives a safe upper bound for the history needed before the training interval.
    maximum_edge_lookback = int(np.max(targets - sources))
    context = maximum_edge_lookback * message_passing_layers
    start = max(0, int(train_positions.min()) - context)
    stop = int(test_positions.max()) + 1

    # Keep edges whose source and target are both inside the local node interval.
    # Subtracting start changes global node identifiers to local identifiers.
    edge_mask = (sources >= start) & (targets >= start) & (targets < stop)
    if edge_mode == "temporal":
        edge_mask &= graph["edge_type"] == EDGE_TEMPORAL
    local_edges = graph["edge_index"][:, edge_mask] - start

    data = Data(
        x=torch.from_numpy(np.asarray(node_features[start:stop], dtype=np.float32)),
        edge_index=torch.from_numpy(local_edges.astype(np.int64, copy=False)),
        y=torch.from_numpy(
            graph["targets"][start:stop].astype(np.int64, copy=False)
        ),
        train_mask=torch.from_numpy(graph_train_mask[start:stop]),
        validation_mask=torch.from_numpy(graph_validation_mask[start:stop]),
        test_mask=torch.from_numpy(graph_test_mask[start:stop]),
    )
    data.global_start = start
    data.global_stop = stop
    return data


# Calculate macro F1 on one partition. Passing all three labels keeps an absent or
# unpredicted anomaly class in the average with score zero.
def masked_macro_f1(logits, targets, mask):
    predictions = logits.argmax(dim=1)[mask].detach().cpu().numpy()
    actual = targets[mask].detach().cpu().numpy()
    score = f1_score(
        actual,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    return float(score)


# Train a GAT or MLP with the same optimizer, loss and early stopping rule. Keeping
# the protocol identical makes architecture the meaningful experimental difference.
def train_neural_baseline(data, config, model_family):
    set_reproducible_seed(config["seed"])
    torch.use_deterministic_algorithms(config.get("deterministic_algorithms", True))
    device = torch.device(config["device"])
    data = data.to(device)

    if model_family == "gat":
        model = GATBaseline(
            input_channels=data.num_node_features,
            hidden_channels=config["hidden_channels"],
            attention_heads=config["attention_heads"],
            dropout=config["dropout"],
        ).to(device)
    else:
        model = MLPBaseline(
            input_channels=data.num_node_features,
            hidden_channels=config["hidden_channels"],
            dropout=config["dropout"],
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    class_weights = compute_class_weights(data.y, data.train_mask).to(device)

    best_state = None
    best_validation = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["maximum_epochs"] + 1):
        # Only training nodes contribute to gradients and parameter updates.
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        training_loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights,
        )
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Validation chooses the checkpoint and stopping epoch. The test mask stays
        # unused until the best validation checkpoint has been restored.
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            validation_loss = F.cross_entropy(
                logits[data.validation_mask],
                data.y[data.validation_mask],
                weight=class_weights,
            )
            validation_f1 = masked_macro_f1(
                logits,
                data.y,
                data.validation_mask,
            )

        history.append(
            {
                "epoch": epoch,
                "training_loss": float(training_loss.item()),
                "validation_loss": float(validation_loss.item()),
                "validation_macro_f1": validation_f1,
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

    # Evaluate the checkpoint selected only by validation, not the final epoch.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits.argmax(dim=1)[data.test_mask].cpu().numpy()
        actual = data.y[data.test_mask].cpu().numpy()

    test_macro_f1 = float(
        f1_score(
            actual,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )
    report = classification_report(
        actual,
        predictions,
        labels=[0, 1, 2],
        target_names=["normal", "up", "down"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predictions, labels=[0, 1, 2]).tolist()

    result = {
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "validation_macro_f1": best_validation,
        "test_macro_f1": test_macro_f1,
        "test_classification_report": report,
        "test_confusion_matrix": matrix,
        "history": history,
    }
    return model, result


# Predict the most common training class without features. This measures whether a
# neural model improves over the simplest class-imbalance baseline.
def evaluate_majority_baseline(data):
    training_targets = data.y[data.train_mask]
    counts = torch.bincount(training_targets, minlength=3)
    majority_class = int(torch.argmax(counts).item())

    validation_actual = data.y[data.validation_mask].cpu().numpy()
    test_actual = data.y[data.test_mask].cpu().numpy()
    validation_predictions = np.full_like(validation_actual, majority_class)
    test_predictions = np.full_like(test_actual, majority_class)

    validation_f1 = float(
        f1_score(
            validation_actual,
            validation_predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )
    test_f1 = float(
        f1_score(
            test_actual,
            test_predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )
    report = classification_report(
        test_actual,
        test_predictions,
        labels=[0, 1, 2],
        target_names=["normal", "up", "down"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_actual,
        test_predictions,
        labels=[0, 1, 2],
    ).tolist()

    return {
        "best_epoch": 0,
        "epochs_trained": 0,
        "validation_macro_f1": validation_f1,
        "test_macro_f1": test_f1,
        "test_classification_report": report,
        "test_confusion_matrix": matrix,
        "history": [],
    }
