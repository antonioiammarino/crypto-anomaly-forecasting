# Build the causal graph used by the independent GAT experiments.

import json
from pathlib import Path

import numpy as np

from crypto_gnn.data.preparation.preprocessing import load_cryptocurrency


EDGE_TEMPORAL = 0
EDGE_VISIBILITY = 1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = PROJECT_ROOT / "dataset crypto" / "cryptos_merged"
SUMMARY_OUTPUT = PROJECT_ROOT / "artifacts" / "btc_graph_summary.json"
GRAPH_SYMBOL = "BTC"
VISIBILITY_WINDOW = 30


# Find non-consecutive Natural Visibility Graph edges. Looking backward from each
# target keeps every edge directed from the past to the future.
def _causal_natural_visibility_edges(values, max_lookback, segment_starts):
    visibility_sources = []
    visibility_targets = []

    for target in range(1, len(values)):
        # The search stops at both the visibility window and the latest data gap.
        # This prevents an edge from using an arbitrarily old or missing period.
        start = max(int(segment_starts[target]), target - max_lookback)
        if start == target:
            continue

        target_value = values[target]
        closest = target - 1
        maximum_slope = values[closest] - target_value

        for source in range(target - 2, start - 1, -1):
            distance = target - source
            slope = (values[source] - target_value) / distance
            if slope > maximum_slope:
                visibility_sources.append(source)
                visibility_targets.append(target)
                maximum_slope = slope

    sources = np.asarray(visibility_sources, dtype=np.int64)
    targets = np.asarray(visibility_targets, dtype=np.int64)
    return sources, targets


# Build one graph from the observed rows of a cryptocurrency. Missing rows remain
# in the original timeline but do not become nodes or connect separate segments.
def build_temporal_visibility_graph(frame, visibility_window=30):
    observation_valid = frame["observation_valid"].to_numpy(dtype=bool)
    source_positions = np.flatnonzero(observation_valid).astype(np.int64)
    observed = frame.iloc[source_positions]
    number_of_nodes = len(observed)

    # Consecutive observations receive the basic temporal edge. Two adjacent rows
    # in the node array may still be separated by a missing hour in the dataset.
    time_differences = observed.index.to_series().diff()
    consecutive = (
        time_differences.to_numpy(dtype="timedelta64[ns]")
        == np.timedelta64(1, "h")
    )
    temporal_targets = np.flatnonzero(consecutive).astype(np.int64)
    temporal_sources = temporal_targets - 1

    # Record where each continuous segment begins. Visibility edges can then be
    # generated without repeatedly searching the original missing observations.
    segment_starts = np.zeros(number_of_nodes, dtype=np.int64)
    current_segment_start = 0
    for node in range(1, number_of_nodes):
        if not consecutive[node]:
            current_segment_start = node
        segment_starts[node] = current_segment_start

    close_values = observed["Close"].to_numpy(dtype=np.float64, copy=True)
    visibility_sources, visibility_targets = _causal_natural_visibility_edges(
        close_values,
        visibility_window,
        segment_starts,
    )

    # Edge types are stored separately so the same graph can support the temporal
    # GAT and the temporal-plus-visibility GAT without rebuilding its nodes.
    edge_index = np.vstack(
        [
            np.concatenate([temporal_sources, visibility_sources]),
            np.concatenate([temporal_targets, visibility_targets]),
        ]
    )
    edge_type = np.concatenate(
        [
            np.full(len(temporal_sources), EDGE_TEMPORAL, dtype=np.int8),
            np.full(len(visibility_sources), EDGE_VISIBILITY, dtype=np.int8),
        ]
    )

    symbols = frame["symbol"].unique()
    graph = {
        "symbol": str(symbols[0]),
        "timestamps": observed.index.copy(),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "targets": observed["target"].to_numpy(dtype=np.int64, copy=True),
        "target_mask": observed["target_valid"].to_numpy(dtype=bool, copy=True),
        "source_positions": source_positions,
        "number_of_nodes": number_of_nodes,
        "number_of_edges": edge_index.shape[1],
    }
    return graph


# Summarize one graph before training. The counts make it easy to compare the new
# graph construction with the graph used by the completed experiments.
def summarize_graph(graph):
    sources = graph["edge_index"][0]
    targets = graph["edge_index"][1]
    lookbacks = targets - sources
    timestamp_differences = graph["timestamps"].to_series().diff()
    contiguous = (
        timestamp_differences.to_numpy(dtype="timedelta64[ns]")
        == np.timedelta64(1, "h")
    )

    summary = {
        "symbol": graph["symbol"],
        "nodes": graph["number_of_nodes"],
        "edges": graph["number_of_edges"],
        "temporal_edges": int(
            np.count_nonzero(graph["edge_type"] == EDGE_TEMPORAL)
        ),
        "visibility_edges": int(
            np.count_nonzero(graph["edge_type"] == EDGE_VISIBILITY)
        ),
        "maximum_edge_lookback_hours": int(lookbacks.max()),
        "mean_edge_lookback_hours": float(lookbacks.mean()),
        "valid_targets": int(graph["target_mask"].sum()),
        "contiguous_segments": int(1 + np.count_nonzero(~contiguous[1:])),
        "start_utc": graph["timestamps"].min().isoformat(),
        "end_utc": graph["timestamps"].max().isoformat(),
        "causal": bool(np.all(sources < targets)),
    }
    return summary


# Build the default BTC graph and save its diagnostic summary.
if __name__ == "__main__":
    frame = load_cryptocurrency(DATASET_DIRECTORY / f"{GRAPH_SYMBOL}.csv")
    graph = build_temporal_visibility_graph(frame, VISIBILITY_WINDOW)
    summary = summarize_graph(graph)

    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {SUMMARY_OUTPUT}")
