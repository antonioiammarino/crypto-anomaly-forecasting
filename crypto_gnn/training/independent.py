# Nested walk-forward experiment for independent MLP and GAT models.

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_gnn.data.features import (
    build_causal_feature_window,
    build_causal_features,
    fit_feature_standardizer,
    standardize_features,
)
from crypto_gnn.data.preprocessing import load_cryptocurrency
from crypto_gnn.graph.temporal import build_temporal_visibility_graph
from crypto_gnn.splits import make_nested_monthly_splits
from crypto_gnn.training.common import (
    evaluate_majority_baseline,
    prepare_fold_data,
    train_neural_baseline,
)
from crypto_gnn.training.config_independent import (
    EXPERIMENT,
    PROTOCOL_NAME,
)
from crypto_gnn.training.search import (
    MAJORITY_CANDIDATE,
    build_candidates,
)


# Convert Path objects when experiment dictionaries are saved as JSON.
def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


# Write through a temporary file so an interrupted process cannot leave a partial
# configuration or progress file.
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


# Save result tables after each selected model so a long experiment retains all
# completed work even if a later fit is interrupted.
def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


# Keep the primary metrics and confusion counts for each outer run. The confusion
# matrix is enough to calculate class-level precision, recall and F1 later.
def outer_result_columns(result):
    report = result["test_classification_report"]
    columns = {
        "validation_macro_f1": result["validation_macro_f1"],
        "test_macro_f1": result["test_macro_f1"],
        "test_accuracy": report["accuracy"],
        "best_epoch": result["best_epoch"],
        "epochs_trained": result["epochs_trained"],
    }

    class_names = ("normal", "up", "down")
    matrix = result["test_confusion_matrix"]
    for actual_index, actual_name in enumerate(class_names):
        for predicted_index, predicted_name in enumerate(class_names):
            column_name = f"actual_{actual_name}_predicted_{predicted_name}"
            columns[column_name] = matrix[actual_index][predicted_index]
    return columns


# Keep data, graphs and features in memory during one execution. Loading each CSV
# and rebuilding each graph for every fit would add unnecessary repeated work.
def create_repository(settings):
    return {
        "settings": settings,
        "frames": {},
        "graphs": {},
        "features": {},
        "nested_splits": {},
        "inner_results": {},
    }


# Load a cryptocurrency only the first time it is requested.
def repository_frame(repository, symbol):
    if symbol not in repository["frames"]:
        settings = repository["settings"]
        path = settings["dataset_directory"] / f"{symbol}.csv"
        repository["frames"][symbol] = load_cryptocurrency(path)
    return repository["frames"][symbol]


# Reuse graphs with the same symbol and visibility window. MLP and temporal GAT use
# a minimal window because they need node alignment but ignore visibility edges.
def repository_graph(repository, symbol, candidate):
    graph_window = candidate["visibility_window"]
    if graph_window is None:
        graph_window = 2
    key = (symbol, graph_window)
    if key not in repository["graphs"]:
        frame = repository_frame(repository, symbol)
        repository["graphs"][key] = build_temporal_visibility_graph(
            frame,
            visibility_window=graph_window,
        )
    return repository["graphs"][key]


# Build raw causal features once per asset and feature family. Standardization is
# repeated because its statistics change with each fold.
def repository_features(repository, symbol, feature_set, graph):
    key = (symbol, feature_set)
    if key not in repository["features"]:
        frame = repository_frame(repository, symbol)
        observed = frame.iloc[graph["source_positions"]]
        repository["features"][key] = build_causal_features(observed, feature_set)
    return repository["features"][key]


# Build nested chronological splits once per asset and index them by outer month.
def repository_splits(repository, symbol):
    if symbol not in repository["nested_splits"]:
        settings = repository["settings"]
        frame = repository_frame(repository, symbol)
        nested = make_nested_monthly_splits(
            frame.index,
            first_outer_test_month=settings["first_outer_test_month"],
            last_outer_test_month=settings["last_outer_test_month"],
            inner_folds=settings["inner_folds"],
            forecast_horizon_hours=settings["forecast_horizon_hours"],
            target_valid=frame["target_valid"].to_numpy(),
        )

        monthly_splits = {}
        for fold in nested:
            month = fold["outer"]["test_start"].strftime("%Y-%m")
            monthly_splits[month] = fold
        repository["nested_splits"][symbol] = monthly_splits
    return repository["nested_splits"][symbol]


def create_progress(experiment_root, total_runs, report_every):
    progress = {
        "total_runs": total_runs,
        "completed_runs": 0,
        "report_every": report_every,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current_run": None,
    }
    save_progress(experiment_root, progress, "running")
    return progress


# Save a compact progress file. Timing fields are operational and are not part of
# the scientific equivalence comparison.
def save_progress(experiment_root, progress, status):
    payload = {
        "protocol": PROTOCOL_NAME,
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_runs": progress["total_runs"],
        "completed_runs": progress["completed_runs"],
        "current_run": progress["current_run"],
    }
    write_json(experiment_root / "progress.json", payload)


# Record one completed fit and periodically print progress during training.
def complete_progress(experiment_root, progress):
    progress["completed_runs"] += 1
    progress["current_run"] = None
    save_progress(experiment_root, progress, "running")

    completed = progress["completed_runs"]
    if completed % progress["report_every"] == 0:
        percent = 100.0 * progress["completed_runs"] / progress["total_runs"]
        print(
            f"Progress: {progress['completed_runs']}/{progress['total_runs']} "
            f"({percent:.2f}%)",
            flush=True,
        )


# Train and evaluate one fold.
def execute_run(
    repository,
    experiment_root,
    role,
    feature_set,
    split,
    symbol,
    candidate,
    seed,
    progress,
):
    # Adjacent outer months reuse some inner months. The same candidate, asset,
    # features and seed give the same score, so train that fold only once.
    if role == "inner_selection":
        key = (
            feature_set,
            split["test_start"],
            symbol,
            candidate["identifier"],
            seed,
        )
        if key in repository["inner_results"]:
            return repository["inner_results"][key]

    progress["current_run"] = {
        "role": role,
        "arm": candidate["experimental_arm"],
        "feature_set": feature_set,
        "month": split["test_start"].strftime("%Y-%m"),
        "symbol": symbol,
        "candidate": candidate["identifier"],
        "seed": seed,
    }
    save_progress(experiment_root, progress, "running")
    graph = repository_graph(repository, symbol, candidate)
    source_positions = graph["source_positions"]
    graph_training_mask = split["train_mask"][source_positions]

    if candidate["architecture"] == "majority":
        node_features = np.zeros((graph["number_of_nodes"], 1), dtype=np.float32)
        data = prepare_fold_data(graph, node_features, split, edge_mode="temporal")
        result = evaluate_majority_baseline(data)
    else:
        raw_features = repository_features(
            repository,
            symbol,
            feature_set,
            graph,
        )
        standardizer = fit_feature_standardizer(
            raw_features,
            graph_training_mask,
        )
        standardized = standardize_features(raw_features, standardizer)

        # The MLP receives an explicit causal window. Missing lag blocks are zero
        # after standardization and have separate availability indicators.
        if candidate["architecture"] == "mlp":
            node_features = build_causal_feature_window(
                standardized,
                graph["source_positions"],
                candidate["temporal_window_hours"],
            )
        else:
            node_features = standardized

        data = prepare_fold_data(
            graph,
            node_features,
            split,
            edge_mode=candidate["edge_mode"],
        )
        settings = repository["settings"]
        training_config = {
            "hidden_channels": candidate["hidden_channels"],
            "attention_heads": candidate["attention_heads"],
            "dropout": candidate["dropout"],
            "learning_rate": candidate["learning_rate"],
            "weight_decay": candidate["weight_decay"],
            "maximum_epochs": settings["maximum_epochs"],
            "patience": settings["patience"],
            "seed": seed,
            "device": settings["device"],
            "deterministic_algorithms": True,
        }
        model_family = candidate["architecture"]
        model, result = train_neural_baseline(data, training_config, model_family)
        del model

    # Inner selection needs only the score used to rank candidates. Outer runs keep
    # the metrics required for the single final results table.
    if role == "inner_selection":
        payload = {"test_macro_f1": result["test_macro_f1"]}
        repository["inner_results"][key] = payload
    else:
        payload = outer_result_columns(result)
    complete_progress(experiment_root, progress)
    return payload


# List the outer test months in their fixed chronological order.
def outer_months(settings):
    months = []
    dates = pd.date_range(
        settings["first_outer_test_month"],
        settings["last_outer_test_month"],
        freq="MS",
    )
    for timestamp in dates:
        months.append(timestamp.strftime("%Y-%m"))
    return months


# Count every unique inner fit, outer neural fit and majority evaluation before the
# experiment starts so the expected workload is recorded with the results.
def describe_plan(settings, candidates_by_arm):
    number_of_outer_months = len(outer_months(settings))
    unique_inner_months = number_of_outer_months + settings["inner_folds"] - 1

    candidate_count = 0
    for candidates in candidates_by_arm.values():
        candidate_count += len(candidates)

    inner_runs = len(settings["feature_sets"])
    inner_runs *= candidate_count
    inner_runs *= unique_inner_months
    inner_runs *= len(settings["symbols"])
    inner_runs *= len(settings["tuning_seeds"])

    outer_runs = len(settings["feature_sets"])
    outer_runs *= len(settings["neural_arms"])
    outer_runs *= number_of_outer_months
    outer_runs *= len(settings["symbols"])
    outer_runs *= len(settings["evaluation_seeds"])

    majority_runs = number_of_outer_months * len(settings["symbols"])
    plan = {
        "inner_neural_runs": inner_runs,
        "outer_neural_runs": outer_runs,
        "outer_majority_evaluations": majority_runs,
        "total_runs": inner_runs + outer_runs + majority_runs,
    }

    print("Nested walk-forward cross-validation")
    print(f"Cryptocurrencies: {len(settings['symbols'])}")
    print(f"Outer months: {number_of_outer_months}")
    print(f"Inner folds per outer month: {settings['inner_folds']}")
    print(f"Estimated total runs: {plan['total_runs']}")
    return plan


# Choose the candidate with highest mean inner score, then lower variability and
# finally its stable identifier. The last two values provide deterministic ties.
def select_best_candidate(scored_candidates):
    selected = scored_candidates[0]
    selected_key = (
        -selected["mean_score"],
        selected["score_std"],
        selected["candidate"]["identifier"],
    )
    for scored in scored_candidates[1:]:
        key = (
            -scored["mean_score"],
            scored["score_std"],
            scored["candidate"]["identifier"],
        )
        if key < selected_key:
            selected = scored
            selected_key = key
    return selected["candidate"]


# Run nested model selection and outer evaluation in chronological order.
def execute_nested_protocol(
    settings,
    candidates_by_arm,
    experiment_root,
    progress,
):
    repository = create_repository(settings)
    selection_rows = []
    outer_rows = []

    for outer_month in outer_months(settings):
        print(f"Starting outer month {outer_month}", flush=True)

        # Majority is evaluated once per asset because it has no random weights.
        for symbol in settings["symbols"]:
            nested = repository_splits(repository, symbol)[outer_month]
            outer_split = nested["outer"]
            result = execute_run(
                repository,
                experiment_root,
                "outer_evaluation",
                "none",
                outer_split,
                symbol,
                MAJORITY_CANDIDATE,
                0,
                progress,
            )
            row = {
                "outer_month": outer_month,
                "experimental_arm": "majority",
                "feature_set": "none",
                "symbol": symbol,
                "seed": 0,
                "selected_candidate_id": MAJORITY_CANDIDATE["identifier"],
            }
            row.update(result)
            outer_rows.append(row)

        for arm in settings["neural_arms"]:
            candidates = candidates_by_arm[arm]
            for feature_set in settings["feature_sets"]:
                scored_candidates = []

                # Every candidate is scored over the same assets, inner months and
                # seeds before the outer test month is opened.
                for candidate in candidates:
                    scores = []
                    for symbol in settings["symbols"]:
                        nested = repository_splits(repository, symbol)[outer_month]
                        for inner_split in nested["inner"]:
                            for seed in settings["tuning_seeds"]:
                                result = execute_run(
                                    repository,
                                    experiment_root,
                                    "inner_selection",
                                    feature_set,
                                    inner_split,
                                    symbol,
                                    candidate,
                                    seed,
                                    progress,
                                )
                                scores.append(float(result["test_macro_f1"]))

                    mean_score = float(np.mean(scores))
                    if len(scores) > 1:
                        score_std = float(np.std(scores, ddof=1))
                    else:
                        score_std = 0.0
                    scored_candidates.append(
                        {
                            "mean_score": mean_score,
                            "score_std": score_std,
                            "candidate": candidate,
                        }
                    )

                    row = {
                        "outer_month": outer_month,
                        "experimental_arm": arm,
                        "feature_set": feature_set,
                        "candidate_id": candidate["identifier"],
                        "mean_inner_macro_f1": mean_score,
                        "std_inner_macro_f1": score_std,
                        "inner_evaluations": len(scores),
                    }
                    selection_rows.append(row)
                    print(
                        f"Mean inner macro F1 | {outer_month} | {arm} | "
                        f"{feature_set} | {candidate['identifier']}: "
                        f"{mean_score:.4f}",
                        flush=True,
                    )

                selected = select_best_candidate(scored_candidates)
                print(
                    f"Selected {outer_month} {arm} {feature_set}: "
                    f"{selected['identifier']}",
                    flush=True,
                )

                # Refit the selected candidate with every evaluation seed. The outer
                # test month has not contributed to candidate selection.
                for symbol in settings["symbols"]:
                    nested = repository_splits(repository, symbol)[outer_month]
                    outer_split = nested["outer"]
                    for seed in settings["evaluation_seeds"]:
                        result = execute_run(
                            repository,
                            experiment_root,
                            "outer_evaluation",
                            feature_set,
                            outer_split,
                            symbol,
                            selected,
                            seed,
                            progress,
                        )
                        row = {
                            "outer_month": outer_month,
                            "experimental_arm": arm,
                            "feature_set": feature_set,
                            "symbol": symbol,
                            "seed": seed,
                            "selected_candidate_id": selected["identifier"],
                        }
                        row.update(result)
                        outer_rows.append(row)

                write_csv(
                    experiment_root / "inner_model_selection.csv",
                    pd.DataFrame(selection_rows),
                )
                write_csv(
                    experiment_root / "outer_results.csv",
                    pd.DataFrame(outer_rows),
                )


# Run the selected configuration and return the experiment output directory.
def run_nested_cv(settings):
    candidates_by_arm = {}
    for arm in settings["neural_arms"]:
        candidates_by_arm[arm] = build_candidates(settings["search_space"], arm)
    plan = describe_plan(settings, candidates_by_arm)

    experiment_root = settings["output_directory"] / PROTOCOL_NAME
    configuration = {
        "protocol": PROTOCOL_NAME,
        "settings": settings,
        "plan": plan,
        "candidates_by_arm": candidates_by_arm,
    }
    write_json(experiment_root / "experiment_config.json", configuration)

    progress = create_progress(
        experiment_root,
        plan["total_runs"],
        settings["progress_report_every"],
    )
    try:
        execute_nested_protocol(
            settings,
            candidates_by_arm,
            experiment_root,
            progress,
        )
    except BaseException:
        save_progress(experiment_root, progress, "failed")
        raise

    save_progress(experiment_root, progress, "completed")
    print(f"Experiment completed: {experiment_root}")
    return experiment_root


# Keep execution explicit and controlled by the configuration file so the same
# recorded settings are used when the experiment is restarted.
if __name__ == "__main__":
    run_nested_cv(EXPERIMENT)
