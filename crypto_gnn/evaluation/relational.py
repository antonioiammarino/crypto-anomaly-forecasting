# Statistical evaluation of relational Panel GAT models and independent baselines.

import json
import pandas as pd

from crypto_gnn.evaluation.statistics import (
    analyze_paired_differences,
    build_case_scores,
    build_paired_differences,
    relational_contrasts,
    summarize_models,
)
from crypto_gnn.training.config_independent import (
    EXPERIMENT as INDEPENDENT_EXPERIMENT,
)
from crypto_gnn.training.config_independent import (
    PROTOCOL_NAME as INDEPENDENT_PROTOCOL,
)
from crypto_gnn.training.config_relational import EXPERIMENT, PANEL_ARMS, PROTOCOL_NAME


ANALYSIS_PROTOCOL = "relational_paired_month_block_bootstrap_v1"


# Save the analysis choices next to the derived results so the report remains
# reproducible without adding timestamps or machine-specific paths.
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


# Write each statistical level to a plain CSV file for direct inspection.
def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# Combine only the OHLC methods needed by the planned relational comparisons.
# Both experiments use the same outer months and cryptocurrencies, so pairing is
# performed on those shared observations after averaging their evaluation seeds.
def evaluate_relational_results(relational_results, independent_results, config):
    relational_cases = build_case_scores(relational_results)
    independent_cases = build_case_scores(independent_results)

    relational_cases = relational_cases[
        relational_cases["experimental_arm"].isin(PANEL_ARMS)
        & relational_cases["feature_set"].eq("ohlc")
    ]
    independent_cases = independent_cases[
        independent_cases["experimental_arm"].isin(
            ("gat_temporal", "mlp_temporal")
        )
        & independent_cases["feature_set"].eq("ohlc")
    ]
    combined = pd.concat([relational_cases, independent_cases], ignore_index=True)

    contrasts = relational_contrasts()
    paired = build_paired_differences(combined, config["metric"], contrasts)
    planned, monthly, sensitivity = analyze_paired_differences(
        paired,
        config,
        contrasts,
    )
    return {
        "model_summary": summarize_models(combined),
        "paired_differences": paired,
        "monthly_differences": monthly,
        "planned_contrasts": planned,
        "block_length_sensitivity": sensitivity,
    }


# Run the complete relational evaluation after both full training experiments are
# available. The independent results supply the two pre-specified baselines.
if __name__ == "__main__":
    relational_directory = EXPERIMENT["output_directory"] / PROTOCOL_NAME
    independent_directory = (
        INDEPENDENT_EXPERIMENT["output_directory"] / INDEPENDENT_PROTOCOL
    )
    relational_results = pd.read_csv(relational_directory / "outer_results.csv")
    independent_results = pd.read_csv(independent_directory / "outer_results.csv")
    statistical_config = EXPERIMENT["statistical_analysis"]
    tables = evaluate_relational_results(
        relational_results,
        independent_results,
        statistical_config,
    )

    output_directory = relational_directory / "statistical_analysis"
    write_csv(output_directory / "model_summary.csv", tables["model_summary"])
    write_csv(
        output_directory / "paired_crypto_month_differences.csv",
        tables["paired_differences"],
    )
    write_csv(
        output_directory / "monthly_paired_differences.csv",
        tables["monthly_differences"],
    )
    write_csv(
        output_directory / "planned_contrasts.csv",
        tables["planned_contrasts"],
    )
    write_csv(
        output_directory / "block_length_sensitivity.csv",
        tables["block_length_sensitivity"],
    )
    write_json(
        output_directory / "analysis_config.json",
        {
            "analysis_protocol": ANALYSIS_PROTOCOL,
            "relational_training_protocol": PROTOCOL_NAME,
            "independent_training_protocol": INDEPENDENT_PROTOCOL,
            "statistical_config": statistical_config,
            "planned_contrasts": relational_contrasts(),
            "interpretation": (
                "Positive differences favour the left-hand method. Two-sided "
                "p-values are Holm-adjusted across all six planned contrasts."
            ),
        },
    )

    print(f"Statistical evaluation completed: {output_directory}")
    print(tables["model_summary"].to_string(index=False))
