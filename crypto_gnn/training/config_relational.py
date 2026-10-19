# Configuration for the complete relational nested walk-forward experiment.

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_NAME = "relational_nested_walk_forward_v1"

ALL_SYMBOLS = (
    "BTC",
    "BTS",
    "DASH",
    "DGB",
    "DOGE",
    "ETH",
    "IOC",
    "LTC",
    "MAID",
    "MONA",
    "NAV",
    "SYS",
    "VTC",
    "XCP",
    "XLM",
    "XMR",
    "XRP",
)

PANEL_ARMS = (
    "panel_gat_temporal",
    "panel_gat_clustered",
    "panel_gat_all",
)


# The clustered arm covers every CARROT window and cluster-count pair before the
# remaining slots explore neural hyperparameters with the same total budget.
SEARCH_SPACE = {
    "clustering_lookback_months": (1, 3, 6),
    "cluster_counts": (3, 5, 7),
    "hidden_channels": (8, 16, 32),
    "attention_heads": (1, 2, 4),
    "dropout": (0.1, 0.3, 0.5),
    "learning_rate": (0.001, 0.003, 0.005),
    "weight_decay": (0.0, 0.0001, 0.0005),
    "candidates_per_arm": 15,
    "search_seed": 2026,
}


STATISTICAL_ANALYSIS = {
    "metric": "macro_f1_mean_over_seeds",
    "confidence_level": 0.95,
    "bootstrap_replicates": 10_000,
    "primary_block_length_months": 3,
    "sensitivity_block_lengths_months": (2, 4),
    "bootstrap_seed": 2026,
    "multiple_testing_correction": "holm",
    "alternative": "two-sided",
    "expected_months": 12,
    "expected_symbols_per_month": 17,
}


EXPERIMENT = {
    "symbols": ALL_SYMBOLS,
    "panel_arms": PANEL_ARMS,
    "feature_set": "ohlc",
    "first_outer_test_month": "2021-01-01",
    "last_outer_test_month": "2021-12-01",
    "inner_folds": 5,
    "forecast_horizon_hours": 1,
    "tuning_seeds": (13, 42, 73),
    "evaluation_seeds": (13, 42, 73, 101, 137),
    "maximum_epochs": 150,
    "patience": 15,
    "device": "cpu",
    "search_space": SEARCH_SPACE,
    "statistical_analysis": STATISTICAL_ANALYSIS,
    "dataset_directory": PROJECT_ROOT / "dataset crypto" / "cryptos_merged",
    "clustering_directory": (
        PROJECT_ROOT
        / "artifacts"
        / "temporal_clustering"
        / "temporal_clustering_v1"
    ),
    "output_directory": PROJECT_ROOT / "artifacts" / "relational_nested_cv",
    "progress_report_every": 5,
}
