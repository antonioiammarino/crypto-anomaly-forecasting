# Temporal stability analysis of the saved DTW partitions.

import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from crypto_gnn.clustering.config_clustering import (
    FIRST_CUTOFF,
    LAST_CUTOFF,
    LOOKBACK_MONTHS,
    NUMBERS_OF_CLUSTERS,
    OUTPUT_DIRECTORY,
)


# Measure how many asset pairs change from together to separate, or the reverse,
# between two partitions. This complements ARI with a directly readable rate.
def pair_disagreement(first, second):
    upper = np.triu_indices(len(first), k=1)
    first_pairs = (first[:, None] == first[None, :])[upper]
    second_pairs = (second[:, None] == second[None, :])[upper]
    return float(np.mean(first_pairs != second_pairs))


# Load every partition once because all stability calculations reuse the same
# aligned labels in several summaries.
def load_partitions(cutoffs):
    partitions = {}
    symbols = None

    for cutoff in cutoffs:
        for lookback_months in LOOKBACK_MONTHS:
            for number_of_clusters in NUMBERS_OF_CLUSTERS:
                path = (
                    OUTPUT_DIRECTORY
                    / f"cutoff_{cutoff}"
                    / f"window_{lookback_months}m"
                    / f"k_{number_of_clusters}.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                if symbols is None:
                    symbols = tuple(sorted(payload["assignments"]))

                labels = []
                for symbol in symbols:
                    labels.append(payload["assignments"][symbol])
                partitions[(lookback_months, number_of_clusters, cutoff)] = {
                    "payload": payload,
                    "labels": np.asarray(labels, dtype=np.int64),
                }

    return partitions, symbols


# Build detailed rows first, then aggregate them. Keeping both levels allows the
# final averages to be traced back to individual months and asset pairs.
def build_analysis_tables(partitions, symbols, cutoffs):
    partition_rows = []
    transition_rows = []
    pair_rows = []

    for lookback_months in LOOKBACK_MONTHS:
        for number_of_clusters in NUMBERS_OF_CLUSTERS:
            labels_by_cutoff = []

            for cutoff in cutoffs:
                entry = partitions[
                    (lookback_months, number_of_clusters, cutoff)
                ]
                payload = entry["payload"]
                labels = entry["labels"]
                sizes = np.bincount(labels, minlength=number_of_clusters)
                labels_by_cutoff.append(labels)

                partition_rows.append(
                    {
                        "cutoff": cutoff,
                        "lookback_months": lookback_months,
                        "number_of_clusters": number_of_clusters,
                        "silhouette": payload["silhouette"],
                        "objective": payload["objective"],
                        "singleton_clusters": int(np.sum(sizes == 1)),
                        "minimum_cluster_size": int(sizes.min()),
                        "maximum_cluster_size": int(sizes.max()),
                        "medoid_symbols": ",".join(payload["medoid_symbols"]),
                    }
                )

            for position in range(1, len(cutoffs)):
                previous = labels_by_cutoff[position - 1]
                current = labels_by_cutoff[position]
                transition_rows.append(
                    {
                        "lookback_months": lookback_months,
                        "number_of_clusters": number_of_clusters,
                        "previous_cutoff": cutoffs[position - 1],
                        "current_cutoff": cutoffs[position],
                        "adjusted_rand_index": adjusted_rand_score(
                            previous,
                            current,
                        ),
                        "pair_disagreement_rate": pair_disagreement(
                            previous,
                            current,
                        ),
                    }
                )

            for left in range(len(symbols)):
                for right in range(left + 1, len(symbols)):
                    months_together = 0
                    for labels in labels_by_cutoff:
                        if labels[left] == labels[right]:
                            months_together += 1
                    pair_rows.append(
                        {
                            "lookback_months": lookback_months,
                            "number_of_clusters": number_of_clusters,
                            "left_symbol": symbols[left],
                            "right_symbol": symbols[right],
                            "coassignment_frequency": (
                                months_together / len(labels_by_cutoff)
                            ),
                            "months_together": months_together,
                            "months": len(labels_by_cutoff),
                        }
                    )

    return (
        pd.DataFrame(partition_rows),
        pd.DataFrame(transition_rows),
        pd.DataFrame(pair_rows),
    )


# Summarize each lookback and cluster-count combination so the stability tradeoff
# can be compared without losing the detailed tables used to calculate it.
def summarize_stability(partition_frame, transition_frame):
    rows = []
    for lookback_months in LOOKBACK_MONTHS:
        for number_of_clusters in NUMBERS_OF_CLUSTERS:
            partitions = partition_frame[
                (partition_frame["lookback_months"] == lookback_months)
                & (partition_frame["number_of_clusters"] == number_of_clusters)
            ]
            transitions = transition_frame[
                (transition_frame["lookback_months"] == lookback_months)
                & (transition_frame["number_of_clusters"] == number_of_clusters)
            ]
            singleton_clusters = partitions["singleton_clusters"]

            rows.append(
                {
                    "lookback_months": lookback_months,
                    "number_of_clusters": number_of_clusters,
                    "mean_silhouette": partitions["silhouette"].mean(),
                    "std_silhouette": partitions["silhouette"].std(),
                    "mean_singleton_clusters": singleton_clusters.mean(),
                    "partitions_with_singletons": (singleton_clusters > 0).mean(),
                    "mean_minimum_cluster_size": partitions[
                        "minimum_cluster_size"
                    ].mean(),
                    "mean_maximum_cluster_size": partitions[
                        "maximum_cluster_size"
                    ].mean(),
                    "mean_consecutive_ari": transitions[
                        "adjusted_rand_index"
                    ].mean(),
                    "median_consecutive_ari": transitions[
                        "adjusted_rand_index"
                    ].median(),
                    "minimum_consecutive_ari": transitions[
                        "adjusted_rand_index"
                    ].min(),
                    "mean_pair_disagreement": transitions[
                        "pair_disagreement_rate"
                    ].mean(),
                }
            )

    return pd.DataFrame(rows)


# Generate the reports after clustering has completed. No model result is used,
# so this analysis cannot influence the causal partitions.
if __name__ == "__main__":
    cutoffs = []
    timestamps = pd.date_range(FIRST_CUTOFF, LAST_CUTOFF, freq="MS")
    for timestamp in timestamps:
        cutoffs.append(timestamp.strftime("%Y-%m"))

    partitions, symbols = load_partitions(cutoffs)
    partition_frame, transition_frame, pair_frame = build_analysis_tables(
        partitions,
        symbols,
        cutoffs,
    )
    stability_frame = summarize_stability(partition_frame, transition_frame)

    partition_frame.to_csv(
        OUTPUT_DIRECTORY / "partition_diagnostics.csv",
        index=False,
    )
    transition_frame.to_csv(
        OUTPUT_DIRECTORY / "consecutive_partition_stability.csv",
        index=False,
    )
    pair_frame.to_csv(
        OUTPUT_DIRECTORY / "pair_coassignment_frequency.csv",
        index=False,
    )
    stability_frame.to_csv(
        OUTPUT_DIRECTORY / "cluster_stability_summary.csv",
        index=False,
    )

    display = stability_frame.copy()
    numeric_columns = display.columns[2:]
    display[numeric_columns] = display[numeric_columns].round(4)
    print(display.to_string(index=False))
    print(f"\nCluster analysis written to: {OUTPUT_DIRECTORY}")
