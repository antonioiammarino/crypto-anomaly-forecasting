# Statistical functions shared by the independent and relational evaluations.

import math

import numpy as np
import pandas as pd


# Define the nine comparisons planned for the independent experiment. Each
# comparison changes one factor while keeping either features or model fixed.
def independent_contrasts():
    contrasts = []
    for feature_set in ("ohlcv", "ohlc"):
        contrasts.append(
            {
                "identifier": f"visibility_minus_temporal__{feature_set}",
                "family": "graph_structure",
                "left": "gat_visibility",
                "right": "gat_temporal",
                "fixed_dimension": "feature_set",
                "fixed_value": feature_set,
            }
        )
        contrasts.append(
            {
                "identifier": f"gat_temporal_minus_mlp_temporal__{feature_set}",
                "family": "neural_architecture",
                "left": "gat_temporal",
                "right": "mlp_temporal",
                "fixed_dimension": "feature_set",
                "fixed_value": feature_set,
            }
        )
        contrasts.append(
            {
                "identifier": f"gat_visibility_minus_mlp_temporal__{feature_set}",
                "family": "neural_architecture",
                "left": "gat_visibility",
                "right": "mlp_temporal",
                "fixed_dimension": "feature_set",
                "fixed_value": feature_set,
            }
        )

    for arm in ("gat_temporal", "gat_visibility", "mlp_temporal"):
        contrasts.append(
            {
                "identifier": f"ohlcv_minus_ohlc__{arm}",
                "family": "feature_ablation",
                "left": "ohlcv",
                "right": "ohlc",
                "fixed_dimension": "experimental_arm",
                "fixed_value": arm,
            }
        )
    return contrasts


# Define the six comparisons planned for the relational experiment. Positive
# differences always favour the method named on the left.
def relational_contrasts():
    comparisons = (
        (
            "clustered_minus_panel_temporal__ohlc",
            "primary_relational_effect",
            "panel_gat_clustered",
            "panel_gat_temporal",
        ),
        (
            "clustered_minus_all__ohlc",
            "relation_specificity",
            "panel_gat_clustered",
            "panel_gat_all",
        ),
        (
            "all_minus_panel_temporal__ohlc",
            "generic_cross_asset_effect",
            "panel_gat_all",
            "panel_gat_temporal",
        ),
        (
            "panel_temporal_minus_independent_gat__ohlc",
            "parameter_sharing_effect",
            "panel_gat_temporal",
            "gat_temporal",
        ),
        (
            "clustered_minus_independent_gat__ohlc",
            "overall_graph_improvement",
            "panel_gat_clustered",
            "gat_temporal",
        ),
        (
            "clustered_minus_mlp_temporal__ohlc",
            "strong_baseline_comparison",
            "panel_gat_clustered",
            "mlp_temporal",
        ),
    )

    contrasts = []
    for identifier, family, left, right in comparisons:
        contrasts.append(
            {
                "identifier": identifier,
                "family": family,
                "left": left,
                "right": right,
                "fixed_dimension": "feature_set",
                "fixed_value": "ohlc",
            }
        )
    return contrasts


# Average repeated seeds before comparing models. The scientific observation is
# one cryptocurrency in one outer month, not one random initialization.
def build_case_scores(outer_results):
    keys = ["outer_month", "experimental_arm", "feature_set", "symbol"]
    case_scores = (
        outer_results.groupby(keys, as_index=False)
        .agg(
            macro_f1_mean_over_seeds=("test_macro_f1", "mean"),
            test_accuracy_mean_over_seeds=("test_accuracy", "mean"),
            std_over_seeds=("test_macro_f1", "std"),
            seeds=("test_macro_f1", "count"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return case_scores


# Give one compact descriptive row per model and feature family. This table is a
# readable overview, while the paired tables below preserve the inferential unit.
def summarize_models(case_scores):
    summary = (
        case_scores.groupby(["experimental_arm", "feature_set"], as_index=False)[
            "macro_f1_mean_over_seeds"
        ]
        .agg(
            mean_macro_f1="mean",
            std_across_crypto_months="std",
            crypto_months="count",
        )
        .sort_values(["experimental_arm", "feature_set"])
        .reset_index(drop=True)
    )
    return summary


# Align both methods on the same cryptocurrency and month before subtraction.
# This removes between-asset and between-month variation from each comparison.
def build_paired_differences(case_scores, metric, contrasts):
    rows = []
    for contrast in contrasts:
        fixed_dimension = contrast["fixed_dimension"]
        fixed_value = contrast["fixed_value"]
        subset = case_scores[case_scores[fixed_dimension] == fixed_value]

        if fixed_dimension == "feature_set":
            varying_dimension = "experimental_arm"
        else:
            varying_dimension = "feature_set"

        wide = subset.pivot(
            index=["outer_month", "symbol"],
            columns=varying_dimension,
            values=metric,
        )
        left = contrast["left"]
        right = contrast["right"]
        paired = wide.dropna(subset=[left, right]).reset_index()
        paired["contrast_id"] = contrast["identifier"]
        paired["family"] = contrast["family"]
        paired["left"] = left
        paired["right"] = right
        paired["fixed_dimension"] = fixed_dimension
        paired["fixed_value"] = fixed_value
        paired["left_score"] = paired[left]
        paired["right_score"] = paired[right]
        paired["delta_left_minus_right"] = paired[left] - paired[right]
        rows.append(
            paired[
                [
                    "contrast_id",
                    "family",
                    "left",
                    "right",
                    "fixed_dimension",
                    "fixed_value",
                    "outer_month",
                    "symbol",
                    "left_score",
                    "right_score",
                    "delta_left_minus_right",
                ]
            ]
        )

    return pd.concat(rows, ignore_index=True)


# Derive a different deterministic random seed for every contrast and block
# length, so repeated analyses reproduce exactly the same bootstrap samples.
def bootstrap_seed(base_seed, contrast_id, block_length):
    entropy = [base_seed, block_length]
    entropy.extend(contrast_id.encode("utf-8"))
    sequence = np.random.SeedSequence(entropy)
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


# Sample consecutive months and wrap December to January. Blocks preserve part
# of the temporal dependence that an ordinary independent bootstrap would lose.
def circular_block_indices(observations, block_length, replicates, seed):
    generator = np.random.default_rng(seed)
    blocks = math.ceil(observations / block_length)
    starts = generator.integers(0, observations, size=(replicates, blocks))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % observations
    return indices.reshape(replicates, -1)[:, :observations]


# Calculate a percentile interval for the observed effect and a two-sided test
# from the same block samples after centring the monthly differences at zero.
def bootstrap_statistics(monthly_differences, contrast_id, block_length, config):
    indices = circular_block_indices(
        len(monthly_differences),
        block_length,
        config["bootstrap_replicates"],
        bootstrap_seed(config["bootstrap_seed"], contrast_id, block_length),
    )
    bootstrap_means = monthly_differences[indices].mean(axis=1)
    alpha = 1.0 - config["confidence_level"]
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    observed = float(monthly_differences.mean())
    centered = monthly_differences - observed
    null_means = centered[indices].mean(axis=1)
    extreme = np.count_nonzero(np.abs(null_means) >= abs(observed))
    p_value = (1.0 + float(extreme)) / (config["bootstrap_replicates"] + 1.0)
    return float(lower), float(upper), p_value


# Apply Holm's step-down correction across every comparison in the declared
# family. The cumulative maximum keeps adjusted p-values monotonic.
def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running_maximum = 0.0
    total = len(values)

    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[original_index])
        running_maximum = max(running_maximum, candidate)
        adjusted_sorted[rank] = running_maximum

    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


# Aggregate the seventeen asset differences before resampling. The bootstrap
# unit is therefore the outer month, which matches the temporal study design.
def analyze_paired_differences(paired, config, contrasts):
    monthly = (
        paired.groupby(["contrast_id", "family", "outer_month"], as_index=False)[
            "delta_left_minus_right"
        ]
        .agg(
            mean_delta_across_symbols="mean",
            std_delta_across_symbols="std",
            symbols="count",
        )
        .sort_values(["contrast_id", "outer_month"])
        .reset_index(drop=True)
    )

    summary_rows = []
    sensitivity_rows = []
    for contrast in contrasts:
        contrast_id = contrast["identifier"]
        selected = monthly[monthly["contrast_id"] == contrast_id].sort_values(
            "outer_month"
        )
        values = selected["mean_delta_across_symbols"].to_numpy(dtype=float)
        lower, upper, raw_p_value = bootstrap_statistics(
            values,
            contrast_id,
            config["primary_block_length_months"],
            config,
        )

        summary_rows.append(
            {
                "contrast_id": contrast_id,
                "family": contrast["family"],
                "left": contrast["left"],
                "right": contrast["right"],
                "fixed_dimension": contrast["fixed_dimension"],
                "fixed_value": contrast["fixed_value"],
                "months": len(values),
                "symbols_per_month": config["expected_symbols_per_month"],
                "paired_crypto_months": (
                    len(values) * config["expected_symbols_per_month"]
                ),
                "mean_monthly_delta": float(values.mean()),
                "median_monthly_delta": float(np.median(values)),
                "std_monthly_delta": float(values.std(ddof=1)),
                "confidence_level": config["confidence_level"],
                "ci_lower": lower,
                "ci_upper": upper,
                "block_length_months": config["primary_block_length_months"],
                "bootstrap_replicates": config["bootstrap_replicates"],
                "raw_two_sided_p_value": raw_p_value,
            }
        )

        # Alternative block lengths show whether the conclusion depends strongly
        # on the three-month primary choice without replacing that main analysis.
        for block_length in config["sensitivity_block_lengths_months"]:
            lower, upper, raw_p_value = bootstrap_statistics(
                values,
                contrast_id,
                block_length,
                config,
            )
            sensitivity_rows.append(
                {
                    "contrast_id": contrast_id,
                    "block_length_months": block_length,
                    "mean_monthly_delta": float(values.mean()),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "raw_two_sided_p_value": raw_p_value,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary["holm_adjusted_p_value"] = holm_adjust(
        summary["raw_two_sided_p_value"].to_numpy()
    )
    summary["reject_after_holm_0_05"] = summary["holm_adjusted_p_value"] < 0.05
    sensitivity = pd.DataFrame(sensitivity_rows)
    return summary, monthly, sensitivity
