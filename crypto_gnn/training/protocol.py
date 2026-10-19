# Freeze the model choices used by the 2022 confirmatory experiment.

import json

import pandas as pd

from crypto_gnn.training.config_confirmatory import (
    CONFIRMATION,
    FEATURE_SETS,
    FROZEN_PROTOCOL_NAME,
    INDEPENDENT_ARMS,
    INDEPENDENT_PROTOCOL_NAME,
    PANEL_ARMS,
    RELATIONAL_PROTOCOL_NAME,
)


# Save a readable protocol because it is inspected before any 2022 result exists.
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    path.write_text(text + "\n", encoding="utf-8")


# Match each readable candidate identifier to its complete hyperparameter payload.
def candidate_catalog(configuration):
    catalog = {}
    for arm, candidates in configuration["candidates_by_arm"].items():
        for candidate in candidates:
            catalog[(arm, candidate["identifier"])] = candidate
    return catalog


# Rank candidates using only inner-validation summaries from the twelve 2021
# outer folds. Averaging those folds gives every development decision equal weight
# and never opens an outer test score.
def rank_candidates(selection, configuration, score_column, group_columns):
    catalog = candidate_catalog(configuration)
    grouped = (
        selection.groupby(group_columns + ["candidate_id"], as_index=False)[
            score_column
        ]
        .agg(
            mean_inner_macro_f1="mean",
            std_across_outer_folds="std",
            outer_folds="count",
        )
    )

    rankings = []
    group_values = grouped[group_columns].drop_duplicates()
    for _, values in group_values.iterrows():
        selected = grouped.copy()
        for column in group_columns:
            selected = selected[selected[column] == values[column]]
        selected = selected.sort_values(
            ["mean_inner_macro_f1", "std_across_outer_folds", "candidate_id"],
            ascending=[False, True, True],
        )

        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            arm = row["experimental_arm"]
            candidate_id = row["candidate_id"]
            result = {
                "experimental_arm": arm,
                "rank": rank,
                "candidate_id": candidate_id,
                "mean_inner_macro_f1": float(row["mean_inner_macro_f1"]),
                "std_across_outer_folds": float(
                    row["std_across_outer_folds"]
                ),
                "outer_folds": int(row["outer_folds"]),
                "candidate": catalog[(arm, candidate_id)],
            }
            if "feature_set" in group_columns:
                result["feature_set"] = row["feature_set"]
            rankings.append(result)
    return rankings


# Keep only the first-ranked candidate for each independent or panel group.
def selected_candidates(independent_rankings, relational_rankings):
    selected = {}
    for row in independent_rankings:
        if row["rank"] == 1:
            key = f"{row['experimental_arm']}__{row['feature_set']}"
            selected[key] = row
    for row in relational_rankings:
        if row["rank"] == 1:
            selected[row["experimental_arm"]] = row
    return selected


# Build the frozen protocol from compact development outputs. The two outer result
# files are intentionally never loaded, so 2021 test scores cannot affect choices.
def build_frozen_protocol(independent_directory, relational_directory):
    independent_config = json.loads(
        (independent_directory / "experiment_config.json").read_text(
            encoding="utf-8"
        )
    )
    relational_config = json.loads(
        (relational_directory / "experiment_config.json").read_text(
            encoding="utf-8"
        )
    )
    independent_selection = pd.read_csv(
        independent_directory / "inner_model_selection.csv"
    )
    relational_selection = pd.read_csv(
        relational_directory / "inner_model_selection.csv"
    )

    independent_rankings = rank_candidates(
        independent_selection,
        independent_config,
        "mean_inner_macro_f1",
        ["experimental_arm", "feature_set"],
    )
    relational_rankings = rank_candidates(
        relational_selection,
        relational_config,
        "mean_inner_asset_macro_f1",
        ["experimental_arm"],
    )

    return {
        "protocol": FROZEN_PROTOCOL_NAME,
        "selection_policy": (
            "For each model and feature group, maximize the mean inner-validation "
            "macro F1 across the twelve 2021 outer folds. Break ties by lower "
            "standard deviation and then candidate identifier."
        ),
        "outer_2021_scores_used_for_candidate_ranking": False,
        "development_choices": {
            "independent_arms": list(INDEPENDENT_ARMS),
            "independent_feature_sets": list(FEATURE_SETS),
            "panel_arms": list(PANEL_ARMS),
            "panel_feature_set": "ohlc",
            "majority_is_descriptive_only": True,
        },
        "confirmation": {
            "first_test_month": CONFIRMATION["first_test_month"],
            "last_test_month": CONFIRMATION["last_test_month"],
            "forecast_horizon_hours": CONFIRMATION["forecast_horizon_hours"],
            "training_months": 5,
            "validation_months": 1,
            "evaluation_seeds": list(CONFIRMATION["evaluation_seeds"]),
            "maximum_epochs": CONFIRMATION["maximum_epochs"],
            "patience": CONFIRMATION["patience"],
            "device": CONFIRMATION["device"],
            "statistical_analysis": CONFIRMATION["statistical_analysis"],
            "holm_families": {
                "phase_a": "nine independent-model contrasts",
                "phase_b": "six relational contrasts",
            },
        },
        "source_experiments": {
            "independent": INDEPENDENT_PROTOCOL_NAME,
            "relational": RELATIONAL_PROTOCOL_NAME,
        },
        "independent_rankings": independent_rankings,
        "relational_rankings": relational_rankings,
        "selected_candidates": selected_candidates(
            independent_rankings,
            relational_rankings,
        ),
    }


# Load the already frozen protocol used by clustering, training and evaluation.
def load_frozen_protocol():
    path = CONFIRMATION["frozen_protocol_directory"] / "frozen_protocol.json"
    return json.loads(path.read_text(encoding="utf-8"))


# Freeze every scientific choice before preparing or evaluating 2022 data.
if __name__ == "__main__":
    protocol = build_frozen_protocol(
        CONFIRMATION["independent_directory"],
        CONFIRMATION["relational_directory"],
    )
    path = CONFIRMATION["frozen_protocol_directory"] / "frozen_protocol.json"
    write_json(path, protocol)

    print(f"Confirmatory protocol frozen: {path.parent}")
    for key, row in sorted(protocol["selected_candidates"].items()):
        print(
            f"{key}: {row['candidate_id']} "
            f"(mean inner macro F1={row['mean_inner_macro_f1']:.4f})"
        )
