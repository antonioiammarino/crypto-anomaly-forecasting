# Complete causal DTW clustering experiment for the development period.

from importlib.metadata import version
import json

import pandas as pd

from crypto_gnn.clustering.config_clustering import (
    DATASET_DIRECTORY,
    DTW_BACKEND,
    FIRST_CUTOFF,
    LAST_CUTOFF,
    LOOKBACK_MONTHS,
    NUMBERS_OF_CLUSTERS,
    OUTPUT_DIRECTORY,
    PROTOCOL_NAME,
)
from crypto_gnn.clustering.dtw import (
    DTW_DEFINITION,
    pairwise_dtw_distance_matrix,
)
from crypto_gnn.clustering.kmedoids import (
    fit_kmedoids,
    serialize_kmedoids_result,
)
from crypto_gnn.clustering.series import (
    CLUSTERING_TRANSFORMATION,
    build_clustering_series,
    clustering_series_metadata,
)
from crypto_gnn.data.preprocessing import load_all_cryptocurrencies


# Save readable JSON because partitions are inspected directly and later loaded by
# relational training without importing the DTW implementation.
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# Describe the full scientific protocol before calculations start. Paths and
# run-time details are not part of the protocol because they do not affect results.
def build_experiment_config():
    return {
        "protocol": PROTOCOL_NAME,
        "first_cutoff": FIRST_CUTOFF,
        "last_cutoff": LAST_CUTOFF,
        "lookback_months": list(LOOKBACK_MONTHS),
        "numbers_of_clusters": list(NUMBERS_OF_CLUSTERS),
        "dtw_backend": DTW_BACKEND,
        "dtw_definition": DTW_DEFINITION,
        "dtw_exact": True,
        "dtw_warping_window": None,
        "normalization": CLUSTERING_TRANSFORMATION,
        "missing_policy": "shared complete timestamps only; no imputation",
        "kmedoids": "deterministic PAM BUILD+SWAP",
        "dtaidistance_version": version("dtaidistance"),
    }


# Materialize all matrices and partitions in one complete execution. Distance
# matrices are scientific inputs for the relational experiment, so they are saved
# but always recomputed when this workflow is started again.
if __name__ == "__main__":
    cutoffs = pd.date_range(FIRST_CUTOFF, LAST_CUTOFF, freq="MS", tz="UTC")
    total_matrices = len(cutoffs) * len(LOOKBACK_MONTHS)

    print("Causal DTW clustering")
    print(f"Cutoffs: {len(cutoffs)}")
    print(f"Lookback windows: {LOOKBACK_MONTHS} months")
    print(f"Cluster counts: {NUMBERS_OF_CLUSTERS}")
    print(f"Distance matrices to calculate: {total_matrices}")

    write_json(
        OUTPUT_DIRECTORY / "experiment_config.json",
        build_experiment_config(),
    )
    datasets = load_all_cryptocurrencies(DATASET_DIRECTORY)
    summary_rows = []
    completed_matrices = 0

    for cutoff in cutoffs:
        print(f"Starting cutoff {cutoff:%Y-%m}")
        for lookback_months in LOOKBACK_MONTHS:
            series = build_clustering_series(
                datasets,
                cutoff=cutoff,
                lookback_months=lookback_months,
            )
            window_directory = (
                OUTPUT_DIRECTORY
                / f"cutoff_{cutoff:%Y-%m}"
                / f"window_{lookback_months}m"
            )
            window_directory.mkdir(parents=True, exist_ok=True)

            distance_matrix = pairwise_dtw_distance_matrix(
                series["values"],
                backend=DTW_BACKEND,
            )
            distance_frame = pd.DataFrame(
                distance_matrix,
                index=series["symbols"],
                columns=series["symbols"],
            )
            distance_frame.to_csv(window_directory / "distance_matrix.csv")

            metadata = clustering_series_metadata(series)
            metadata.update(
                {
                    "dtw_backend": DTW_BACKEND,
                    "dtw_definition": DTW_DEFINITION,
                    "dtw_exact": True,
                    "dtw_warping_window": None,
                    "distance_matrix_file": "distance_matrix.csv",
                }
            )
            write_json(window_directory / "metadata.json", metadata)

            # One DTW matrix supports all cluster counts because k-medoids changes
            # only the partition, not the pairwise distances.
            for number_of_clusters in NUMBERS_OF_CLUSTERS:
                result = fit_kmedoids(
                    distance_matrix,
                    number_of_clusters=number_of_clusters,
                )
                partition = {
                    "protocol": PROTOCOL_NAME,
                    "cutoff": cutoff.isoformat(),
                    "lookback_months": lookback_months,
                }
                partition.update(
                    serialize_kmedoids_result(result, series["symbols"])
                )
                write_json(
                    window_directory / f"k_{number_of_clusters}.json",
                    partition,
                )

                cluster_parts = []
                for members in partition["clusters"].values():
                    cluster_parts.append(",".join(members))
                cluster_text = " | ".join(cluster_parts)
                summary_rows.append(
                    {
                        "cutoff": f"{cutoff:%Y-%m}",
                        "lookback_months": lookback_months,
                        "number_of_clusters": number_of_clusters,
                        "complete_hours": series["complete_hours"],
                        "discarded_incomplete_hours": (
                            series["candidate_hours"] - series["complete_hours"]
                        ),
                        "medoids": ",".join(partition["medoid_symbols"]),
                        "objective": result["objective"],
                        "silhouette": result["silhouette"],
                        "clusters": cluster_text,
                    }
                )
                print(
                    f"  window {lookback_months}m, k={number_of_clusters}, "
                    f"silhouette={result['silhouette']:.4f}"
                )

            completed_matrices += 1
            print(f"Progress: {completed_matrices}/{total_matrices}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIRECTORY / "cluster_summary.csv", index=False)
    print(f"Clustering completed: {OUTPUT_DIRECTORY}")
