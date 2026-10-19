# Statistical evaluation of the independent MLP and GAT experiment.

import json
import pandas as pd

from crypto_gnn.evaluation.statistics import (
    analyze_paired_differences,
    build_case_scores,
    build_paired_differences,
    independent_contrasts,
    summarize_models,
)
from crypto_gnn.training.config_independent import EXPERIMENT, PROTOCOL_NAME


ANALYSIS_PROTOCOL = "paired_month_block_bootstrap_v1"


# Save JSON in a stable readable form because it documents the choices used to
# produce the statistical tables.
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


# Save derived tables in CSV so they can be inspected without Python and used
# directly in the written report.
def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# Produce every independent evaluation table from the single seed-level result
# table. Keeping this calculation here makes the training output remain compact.
def evaluate_independent_results(outer_results, statistical_config):
    case_scores = build_case_scores(outer_results)
    contrasts = independent_contrasts()
    paired = build_paired_differences(
        case_scores,
        statistical_config["metric"],
        contrasts,
    )
    planned, monthly, sensitivity = analyze_paired_differences(
        paired,
        statistical_config,
        contrasts,
    )
    return {
        "model_summary": summarize_models(case_scores),
        "paired_differences": paired,
        "monthly_differences": monthly,
        "planned_contrasts": planned,
        "block_length_sensitivity": sensitivity,
    }


# Run the evaluation using the complete independent experiment in its fixed
# output directory. No mode switch is needed because only full results are public.
if __name__ == "__main__":
    experiment_directory = EXPERIMENT["output_directory"] / PROTOCOL_NAME
    outer_results = pd.read_csv(experiment_directory / "outer_results.csv")
    statistical_config = EXPERIMENT["statistical_analysis"]
    tables = evaluate_independent_results(outer_results, statistical_config)

    output_directory = experiment_directory / "statistical_analysis"
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
            "training_protocol": PROTOCOL_NAME,
            "statistical_config": statistical_config,
            "planned_contrasts": independent_contrasts(),
            "interpretation": (
                "Positive differences favour the left-hand method. Two-sided "
                "p-values are Holm-adjusted across all nine planned contrasts."
            ),
        },
    )

    print(f"Statistical evaluation completed: {output_directory}")
    print(tables["model_summary"].to_string(index=False))
