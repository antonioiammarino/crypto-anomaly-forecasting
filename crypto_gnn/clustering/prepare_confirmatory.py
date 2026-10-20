# Causal DTW clustering required by the frozen 2022 Panel GAT candidate.

import json
from importlib.metadata import version

import pandas as pd

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
from crypto_gnn.data.preparation.preprocessing import load_all_cryptocurrencies
from crypto_gnn.training.config_confirmatory import CONFIRMATION
from crypto_gnn.training.protocol import load_frozen_protocol


PROTOCOL_NAME = "confirmatory_clustering_2022_v1"
DTW_BACKEND = "dtaidistance"


# Save partition metadata in a readable format because training loads these files
# without importing the native DTW library.
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


# Recalculate the twelve matrices in one complete execution. Saved matrices are
# scientific outputs and every execution recreates them from the raw observations.
if __name__ == "__main__":
    frozen = load_frozen_protocol()
    selected = frozen["selected_candidates"]["panel_gat_clustered"]["candidate"]
    lookback_months = int(selected["clustering_lookback_months"])
    number_of_clusters = int(selected["number_of_clusters"])
    output_directory = CONFIRMATION["frozen_protocol_directory"] / "clusters"
    cutoffs = pd.date_range(
        CONFIRMATION["first_test_month"],
        CONFIRMATION["last_test_month"],
        freq="MS",
        tz="UTC",
    )

    print("Frozen causal DTW clustering for 2022")
    print(f"Selected candidate: {selected['identifier']}")
    print(
        f"Topology: {lookback_months}-month window, "
        f"{number_of_clusters} clusters"
    )
    print(f"Cutoffs: {len(cutoffs)}")

    datasets = load_all_cryptocurrencies(CONFIRMATION["dataset_directory"])
    summary_rows = []
    for position, cutoff in enumerate(cutoffs, start=1):
        series = build_clustering_series(
            datasets,
            cutoff,
            lookback_months,
        )
        distance_matrix = pairwise_dtw_distance_matrix(
            series["values"],
            backend=DTW_BACKEND,
        )
        partition = fit_kmedoids(distance_matrix, number_of_clusters)
        serialized = serialize_kmedoids_result(partition, series["symbols"])

        window_directory = (
            output_directory
            / f"cutoff_{cutoff:%Y-%m}"
            / f"window_{lookback_months}m"
        )
        window_directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            distance_matrix,
            index=series["symbols"],
            columns=series["symbols"],
        ).to_csv(window_directory / "distance_matrix.csv")

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

        partition_payload = {
            "protocol": PROTOCOL_NAME,
            "cutoff": cutoff.isoformat(),
            "lookback_months": lookback_months,
        }
        partition_payload.update(serialized)
        write_json(
            window_directory / f"k_{number_of_clusters}.json",
            partition_payload,
        )

        summary_rows.append(
            {
                "cutoff": f"{cutoff:%Y-%m}",
                "lookback_months": lookback_months,
                "number_of_clusters": number_of_clusters,
                "complete_hours": series["complete_hours"],
                "discarded_incomplete_hours": (
                    series["candidate_hours"] - series["complete_hours"]
                ),
                "medoids": ",".join(serialized["medoid_symbols"]),
                "objective": partition["objective"],
                "silhouette": partition["silhouette"],
            }
        )
        print(
            f"{cutoff:%Y-%m}: silhouette={partition['silhouette']:.4f} "
            f"({position}/{len(cutoffs)})"
        )

    pd.DataFrame(summary_rows).to_csv(
        output_directory / "cluster_summary.csv",
        index=False,
    )
    write_json(
        output_directory / "clustering_config.json",
        {
            "protocol": PROTOCOL_NAME,
            "frozen_protocol": frozen["protocol"],
            "selected_candidate_id": selected["identifier"],
            "lookback_months": lookback_months,
            "number_of_clusters": number_of_clusters,
            "first_cutoff": cutoffs[0].isoformat(),
            "last_cutoff": cutoffs[-1].isoformat(),
            "dtw_backend": DTW_BACKEND,
            "dtaidistance_version": version("dtaidistance"),
            "dtw_definition": DTW_DEFINITION,
            "normalization": CLUSTERING_TRANSFORMATION,
            "missing_policy": "shared complete timestamps only; no imputation",
            "kmedoids": "deterministic PAM BUILD+SWAP",
        },
    )
    print(f"Confirmatory clustering completed: {output_directory}")
