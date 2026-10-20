# Plot a short price interval together with its temporal and visibility edges.

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_gnn.data.preparation.preprocessing import load_cryptocurrency
from crypto_gnn.graph.temporal import (
    EDGE_TEMPORAL,
    EDGE_VISIBILITY,
    build_temporal_visibility_graph,
)


DATASET_DIRECTORY = PROJECT_ROOT / "dataset crypto" / "cryptos_merged"
OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "figures" / "btc_visibility_graph_sample.png"
)
SYMBOL = "BTC"
START = "2021-01-08T00:00:00Z"
HOURS = 30


# Draw graph edges over Close prices. A short interval is used because showing the
# complete graph would hide the difference between temporal and visibility edges.
def plot_graph_sample(frame, graph, output_path):
    timestamps = frame.index.to_pydatetime()
    close = frame["Close"].to_numpy()
    sources = graph["edge_index"][0]
    targets = graph["edge_index"][1]

    figure, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)

    # Visibility edges are drawn first and lightly so the price series stays clear.
    for edge in range(graph["number_of_edges"]):
        if graph["edge_type"][edge] == EDGE_VISIBILITY:
            source = sources[edge]
            target = targets[edge]
            axis.plot(
                [timestamps[source], timestamps[target]],
                [close[source], close[target]],
                color="#3b82f6",
                alpha=0.20,
                linewidth=0.8,
                zorder=1,
            )

    # Consecutive temporal edges form the main chronological backbone.
    for edge in range(graph["number_of_edges"]):
        if graph["edge_type"][edge] == EDGE_TEMPORAL:
            source = sources[edge]
            target = targets[edge]
            axis.plot(
                [timestamps[source], timestamps[target]],
                [close[source], close[target]],
                color="#64748b",
                alpha=0.85,
                linewidth=1.4,
                zorder=2,
            )

    colors = np.asarray(["#475569", "#16a34a", "#dc2626"])
    axis.scatter(
        timestamps,
        close,
        c=colors[graph["targets"]],
        s=34,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    axis.set_title(
        f"{graph['symbol']}: causal temporal and visibility graph ({len(frame)} hours)",
        fontsize=14,
        weight="bold",
    )
    axis.set_xlabel("Timestamp UTC")
    axis.set_ylabel("Close")
    axis.grid(axis="y", color="#cbd5e1", alpha=0.45, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    formatter = mdates.DateFormatter("%d %b\n%H:%M", tz=frame.index.tz)
    axis.xaxis.set_major_formatter(formatter)

    legend_items = [
        plt.Line2D([0], [0], color="#64748b", lw=1.5, label="Temporal edge"),
        plt.Line2D(
            [0],
            [0],
            color="#3b82f6",
            lw=1.5,
            alpha=0.5,
            label="Visibility edge",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#475569",
            label="Normal",
            markersize=7,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#16a34a",
            label="Up",
            markersize=7,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#dc2626",
            label="Down",
            markersize=7,
        ),
    ]
    axis.legend(handles=legend_items, ncol=5, loc="upper center", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


# Build and plot the fixed BTC interval used in the project documentation.
if __name__ == "__main__":
    frame = load_cryptocurrency(DATASET_DIRECTORY / f"{SYMBOL}.csv")
    start = pd.Timestamp(START)
    stop = start + pd.DateOffset(hours=HOURS)
    sample = frame.loc[(frame.index >= start) & (frame.index < stop)].copy()
    graph = build_temporal_visibility_graph(sample, visibility_window=HOURS)
    plot_graph_sample(sample, graph, OUTPUT_PATH)
    print(OUTPUT_PATH)
