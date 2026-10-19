# Configuration for the complete development-period clustering experiment.

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_NAME = "temporal_clustering_v1"
DATASET_DIRECTORY = PROJECT_ROOT / "dataset crypto" / "cryptos_merged"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "temporal_clustering" / PROTOCOL_NAME

FIRST_CUTOFF = "2020-08-01"
LAST_CUTOFF = "2021-12-01"
LOOKBACK_MONTHS = (1, 3, 6)
NUMBERS_OF_CLUSTERS = (3, 5, 7)
DTW_BACKEND = "dtaidistance"
