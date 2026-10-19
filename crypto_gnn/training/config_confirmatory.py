# Configuration for the frozen 2022 confirmatory experiment.

from pathlib import Path

from crypto_gnn.training.config_independent import (
    ALL_SYMBOLS,
    STATISTICAL_ANALYSIS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FROZEN_PROTOCOL_NAME = "confirmatory_protocol_2022_v1"
CONFIRMATORY_PROTOCOL_NAME = "confirmatory_walk_forward_2022_v1"
INDEPENDENT_PROTOCOL_NAME = "nested_walk_forward_v6"
RELATIONAL_PROTOCOL_NAME = "relational_nested_walk_forward_v1"

INDEPENDENT_ARMS = (
    "gat_temporal",
    "gat_visibility",
    "mlp_temporal",
)
FEATURE_SETS = ("ohlcv", "ohlc")
PANEL_ARMS = (
    "panel_gat_temporal",
    "panel_gat_clustered",
    "panel_gat_all",
)

CONFIRMATION = {
    "symbols": ALL_SYMBOLS,
    "first_test_month": "2022-01-01",
    "last_test_month": "2022-12-01",
    "forecast_horizon_hours": 1,
    "evaluation_seeds": (13, 42, 73, 101, 137),
    "maximum_epochs": 150,
    "patience": 15,
    "device": "cpu",
    "statistical_analysis": STATISTICAL_ANALYSIS,
    "dataset_directory": PROJECT_ROOT / "dataset crypto" / "cryptos_merged",
    "independent_directory": (
        PROJECT_ROOT
        / "artifacts"
        / "nested_cv"
        / INDEPENDENT_PROTOCOL_NAME
    ),
    "relational_directory": (
        PROJECT_ROOT
        / "artifacts"
        / "relational_nested_cv"
        / RELATIONAL_PROTOCOL_NAME
    ),
    "frozen_protocol_directory": (
        PROJECT_ROOT
        / "artifacts"
        / "confirmatory_protocol"
        / FROZEN_PROTOCOL_NAME
    ),
    "output_directory": (
        PROJECT_ROOT
        / "artifacts"
        / "confirmatory_test"
        / CONFIRMATORY_PROTOCOL_NAME
    ),
    "progress_report_every": 25,
}
