# Statistical evaluation of the frozen 2022 confirmation.

import json

import pandas as pd

from crypto_gnn.evaluation.statistics import (
    analyze_paired_differences,
    build_case_scores,
    build_paired_differences,
    independent_contrasts,
    relational_contrasts,
    summarize_models,
)
from crypto_gnn.training.config_confirmatory import (
    CONFIRMATION,
    CONFIRMATORY_PROTOCOL_NAME,
)
from crypto_gnn.training.protocol import load_frozen_protocol


ANALYSIS_PROTOCOL = "confirmatory_paired_month_block_bootstrap_2022_v1"


# Save the declared statistical families next to their result tables.
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


# Keep every analysis level in a plain CSV file for direct inspection.
def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# Analyze one pre-specified Holm family independently from the other family.
def analyze_family(case_scores, config, contrasts):
    paired = build_paired_differences(case_scores, config["metric"], contrasts)
    planned, monthly, sensitivity = analyze_paired_differences(
        paired,
        config,
        contrasts,
    )
    return {
        "paired_differences": paired,
        "monthly_differences": monthly,
        "planned_contrasts": planned,
        "block_length_sensitivity": sensitivity,
    }


# Build both confirmatory families after averaging random seeds. Majority remains
# descriptive and is not added to either multiple-testing family.
def evaluate_confirmatory_results(outer_results, config):
    case_scores = build_case_scores(outer_results)
    independent_cases = case_scores[
        case_scores["experimental_arm"].isin(
            ("gat_temporal", "gat_visibility", "mlp_temporal")
        )
        & case_scores["feature_set"].isin(("ohlcv", "ohlc"))
    ]
    relational_cases = case_scores[
        case_scores["experimental_arm"].isin(
            (
                "panel_gat_temporal",
                "panel_gat_clustered",
                "panel_gat_all",
                "gat_temporal",
                "mlp_temporal",
            )
        )
        & case_scores["feature_set"].eq("ohlc")
    ]

    majority = case_scores[case_scores["experimental_arm"] == "majority"][
        ["outer_month", "symbol", "macro_f1_mean_over_seeds"]
    ].rename(columns={"macro_f1_mean_over_seeds": "majority_macro_f1"})
    descriptive = case_scores[
        case_scores["experimental_arm"] != "majority"
    ].merge(majority, on=["outer_month", "symbol"], how="inner")
    descriptive["delta_model_minus_majority"] = (
        descriptive["macro_f1_mean_over_seeds"]
        - descriptive["majority_macro_f1"]
    )

    return {
        "model_summary": summarize_models(case_scores),
        "phase_a": analyze_family(
            independent_cases,
            config,
            independent_contrasts(),
        ),
        "phase_b": analyze_family(
            relational_cases,
            config,
            relational_contrasts(),
        ),
        "descriptive_model_minus_majority": descriptive,
    }


# Evaluate the complete 2022 result table using the choices frozen before testing.
if __name__ == "__main__":
    frozen = load_frozen_protocol()
    config = frozen["confirmation"]["statistical_analysis"]
    outer_results = pd.read_csv(CONFIRMATION["output_directory"] / "outer_results.csv")
    tables = evaluate_confirmatory_results(outer_results, config)
    output_directory = CONFIRMATION["output_directory"] / "statistical_analysis"

    write_csv(output_directory / "model_summary.csv", tables["model_summary"])
    write_csv(
        output_directory / "descriptive_model_minus_majority.csv",
        tables["descriptive_model_minus_majority"],
    )
    for family_name in ("phase_a", "phase_b"):
        family = tables[family_name]
        family_directory = output_directory / f"{family_name}_family"
        write_csv(
            family_directory / "paired_crypto_month_differences.csv",
            family["paired_differences"],
        )
        write_csv(
            family_directory / "monthly_paired_differences.csv",
            family["monthly_differences"],
        )
        write_csv(
            family_directory / "planned_contrasts.csv",
            family["planned_contrasts"],
        )
        write_csv(
            family_directory / "block_length_sensitivity.csv",
            family["block_length_sensitivity"],
        )

    write_json(
        output_directory / "analysis_config.json",
        {
            "analysis_protocol": ANALYSIS_PROTOCOL,
            "training_protocol": CONFIRMATORY_PROTOCOL_NAME,
            "statistical_config": config,
            "holm_correction_policy": (
                "Holm correction is applied separately to the nine independent "
                "contrasts and the six relational contrasts. Majority comparisons "
                "are descriptive only."
            ),
            "phase_a_contrasts": independent_contrasts(),
            "phase_b_contrasts": relational_contrasts(),
        },
    )
    print(f"Confirmatory evaluation completed: {output_directory}")
    print(tables["model_summary"].to_string(index=False))
