# Multi-asset graph with temporal and same-hour relational edges.

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_gnn.data.preprocessing import load_all_cryptocurrencies
from crypto_gnn.graph.temporal import EDGE_TEMPORAL


EDGE_RELATIONAL = 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = PROJECT_ROOT / "dataset crypto" / "cryptos_merged"
CLUSTERING_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "temporal_clustering"
    / "temporal_clustering_v1"
)
DIAGNOSTIC_OUTPUT = PROJECT_ROOT / "artifacts" / "panel_graph_diagnostic_2021_01.json"


# Translate each experimental topology into one cluster label per asset. Distinct
# labels create no cross-asset edges, while one shared label creates all of them.
def cluster_labels_for_mode(symbols, relation_mode, cluster_assignments=None):
    if relation_mode == "none":
        return np.arange(len(symbols), dtype=np.int64)
    if relation_mode == "all":
        return np.zeros(len(symbols), dtype=np.int64)

    labels = []
    for symbol in symbols:
        labels.append(cluster_assignments[symbol])
    return np.asarray(labels, dtype=np.int64)


# Build nodes only for real observations. Temporal edges stop at gaps, and
# relational edges connect different assets at the same observed hour in both
# directions because cross-asset information has no natural direction.
def build_panel_graph(datasets, relation_mode, cluster_assignments=None):
    symbols = tuple(sorted(datasets))
    cluster_labels = cluster_labels_for_mode(
        symbols,
        relation_mode,
        cluster_assignments,
    )
    reference_index = datasets[symbols[0]].index

    observation_columns = []
    target_columns = []
    target_mask_columns = []
    for symbol in symbols:
        frame = datasets[symbol]
        observation_columns.append(frame["observation_valid"].to_numpy(dtype=bool))
        target_columns.append(frame["target"].to_numpy(dtype=np.int64))
        target_mask_columns.append(frame["target_valid"].to_numpy(dtype=bool))

    observations = np.column_stack(observation_columns)
    target_matrix = np.column_stack(target_columns)
    target_mask_matrix = np.column_stack(target_mask_columns)
    time_positions, symbol_indices = np.nonzero(observations)

    # This inverse table converts a pair (hour, asset) into its node identifier.
    # Missing observations keep -1 and therefore cannot become edge endpoints.
    node_ids = np.full(observations.shape, -1, dtype=np.int64)
    node_ids[time_positions, symbol_indices] = np.arange(
        len(time_positions),
        dtype=np.int64,
    )

    temporal_sources = []
    temporal_destinations = []
    for symbol_index in range(len(symbols)):
        consecutive = observations[:-1, symbol_index] & observations[1:, symbol_index]
        destination_positions = np.flatnonzero(consecutive) + 1
        temporal_sources.append(node_ids[destination_positions - 1, symbol_index])
        temporal_destinations.append(node_ids[destination_positions, symbol_index])

    relational_sources = []
    relational_destinations = []
    if relation_mode != "none":
        for source_symbol in range(len(symbols)):
            for destination_symbol in range(len(symbols)):
                if source_symbol == destination_symbol:
                    continue
                if cluster_labels[source_symbol] != cluster_labels[destination_symbol]:
                    continue

                both_observed = (
                    observations[:, source_symbol]
                    & observations[:, destination_symbol]
                )
                positions = np.flatnonzero(both_observed)
                relational_sources.append(node_ids[positions, source_symbol])
                relational_destinations.append(node_ids[positions, destination_symbol])

    temporal_sources = np.concatenate(temporal_sources)
    temporal_destinations = np.concatenate(temporal_destinations)
    if relational_sources:
        relational_sources = np.concatenate(relational_sources)
        relational_destinations = np.concatenate(relational_destinations)
    else:
        relational_sources = np.empty(0, dtype=np.int64)
        relational_destinations = np.empty(0, dtype=np.int64)

    edge_index = np.vstack(
        [
            np.concatenate([temporal_sources, relational_sources]),
            np.concatenate([temporal_destinations, relational_destinations]),
        ]
    )
    edge_type = np.concatenate(
        [
            np.full(len(temporal_sources), EDGE_TEMPORAL, dtype=np.int8),
            np.full(len(relational_sources), EDGE_RELATIONAL, dtype=np.int8),
        ]
    )

    return {
        "symbols": symbols,
        "relation_mode": relation_mode,
        "cluster_labels": cluster_labels,
        "timestamps": pd.DatetimeIndex(reference_index[time_positions]),
        "source_positions": time_positions.astype(np.int64, copy=False),
        "node_symbol_indices": symbol_indices.astype(np.int64, copy=False),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "targets": target_matrix[time_positions, symbol_indices],
        "target_mask": target_mask_matrix[time_positions, symbol_indices],
        "number_of_nodes": len(time_positions),
        "number_of_edges": edge_index.shape[1],
        "number_of_temporal_edges": len(temporal_sources),
        "number_of_relational_edges": len(relational_sources),
    }


# Inspect the three graph topologies on the same real seven-month window. This
# records how many additional messages clustered and all-to-all relations create.
if __name__ == "__main__":
    cutoff = pd.Timestamp("2021-01-01", tz="UTC")
    graph_start = cutoff - pd.DateOffset(months=6)
    graph_end = cutoff + pd.DateOffset(months=1)
    cluster_window_months = 6
    number_of_clusters = 3

    partition_path = (
        CLUSTERING_DIRECTORY
        / f"cutoff_{cutoff:%Y-%m}"
        / f"window_{cluster_window_months}m"
        / f"k_{number_of_clusters}.json"
    )
    partition = json.loads(partition_path.read_text(encoding="utf-8"))

    all_datasets = load_all_cryptocurrencies(DATASET_DIRECTORY)
    datasets = {}
    for symbol, frame in all_datasets.items():
        datasets[symbol] = frame.loc[
            (frame.index >= graph_start) & (frame.index < graph_end)
        ]

    rows = []
    for relation_mode in ("none", "clustered", "all"):
        assignments = None
        if relation_mode == "clustered":
            assignments = partition["assignments"]
        graph = build_panel_graph(datasets, relation_mode, assignments)
        rows.append(
            {
                "relation_mode": relation_mode,
                "nodes": graph["number_of_nodes"],
                "temporal_edges": graph["number_of_temporal_edges"],
                "relational_edges": graph["number_of_relational_edges"],
                "total_edges": graph["number_of_edges"],
                "mean_relational_in_degree": (
                    graph["number_of_relational_edges"] / graph["number_of_nodes"]
                ),
            }
        )
        del graph
        gc.collect()

    payload = {
        "cutoff": cutoff.isoformat(),
        "graph_start": graph_start.isoformat(),
        "graph_end": graph_end.isoformat(),
        "cluster_window_months": cluster_window_months,
        "number_of_clusters": number_of_clusters,
        "partition_file": str(partition_path.relative_to(PROJECT_ROOT)),
        "clusters": partition["clusters"],
        "graphs": rows,
    }
    DIAGNOSTIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nPanel graph summary written to: {DIAGNOSTIC_OUTPUT}")
