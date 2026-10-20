# Nested walk-forward experiment for the three multi-asset Panel GAT topologies.

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_gnn.data.preprocessing import load_all_cryptocurrencies
from crypto_gnn.graph.panel import build_panel_graph
from crypto_gnn.splits import make_nested_monthly_splits
from crypto_gnn.training.config_relational import (
    EXPERIMENT,
    PROTOCOL_NAME,
)
from crypto_gnn.training.panel import (
    build_standardized_panel_features,
    prepare_panel_fold_data,
    train_panel_gat,
)


# Convert paths when experiment settings are written to JSON.
def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


# Write compact configuration and progress files safely during the long experiment.
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


# Save the two result tables after every selected arm so completed outer fits are
# retained if a later model fails.
def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


# Build the neural hyperparameter combinations in a fixed explicit order. The
# seeded sample depends on this order and must remain reproducible.
def complete_neural_space(search_space):
    configurations = []
    for hidden_channels in search_space["hidden_channels"]:
        for attention_heads in search_space["attention_heads"]:
            for dropout in search_space["dropout"]:
                for learning_rate in search_space["learning_rate"]:
                    for weight_decay in search_space["weight_decay"]:
                        configurations.append(
                            {
                                "hidden_channels": hidden_channels,
                                "attention_heads": attention_heads,
                                "dropout": dropout,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                            }
                        )
    return configurations


# Create fifteen candidates per arm. The clustered defaults cover all nine DTW
# topologies, while every arm uses the same neural reference configuration.
def build_candidates(search_space, experimental_arm):
    default_neural = {
        "hidden_channels": 16,
        "attention_heads": 4,
        "dropout": 0.3,
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
    }
    neural_space = complete_neural_space(search_space)
    payloads = []
    complete_space = []

    if experimental_arm == "panel_gat_clustered":
        topologies = []
        for lookback in search_space["clustering_lookback_months"]:
            for clusters in search_space["cluster_counts"]:
                topologies.append((lookback, clusters))

        for lookback, clusters in topologies:
            candidate = {
                "experimental_arm": experimental_arm,
                "relation_mode": "clustered",
                "clustering_lookback_months": lookback,
                "number_of_clusters": clusters,
            }
            candidate.update(default_neural)
            payloads.append(candidate)

        for lookback, clusters in topologies:
            for neural in neural_space:
                candidate = {
                    "experimental_arm": experimental_arm,
                    "relation_mode": "clustered",
                    "clustering_lookback_months": lookback,
                    "number_of_clusters": clusters,
                }
                candidate.update(neural)
                complete_space.append(candidate)
    else:
        relation_mode = "none"
        if experimental_arm == "panel_gat_all":
            relation_mode = "all"

        candidate = {
            "experimental_arm": experimental_arm,
            "relation_mode": relation_mode,
            "clustering_lookback_months": None,
            "number_of_clusters": None,
        }
        candidate.update(default_neural)
        payloads.append(candidate)

        for neural in neural_space:
            candidate = {
                "experimental_arm": experimental_arm,
                "relation_mode": relation_mode,
                "clustering_lookback_months": None,
                "number_of_clusters": None,
            }
            candidate.update(neural)
            complete_space.append(candidate)

    remaining = []
    for candidate in complete_space:
        if candidate not in payloads:
            remaining.append(candidate)

    additional = search_space["candidates_per_arm"] - len(payloads)
    generator = random.Random(f"{search_space['search_seed']}:{experimental_arm}")
    payloads.extend(generator.sample(remaining, additional))

    candidates = []
    for number, payload in enumerate(payloads, start=1):
        candidate = {"identifier": f"{experimental_arm}_{number}"}
        candidate.update(payload)
        candidates.append(candidate)
    return candidates


# Return scientific candidate fields without duplicating the identifier column.
def candidate_payload(candidate):
    payload = {}
    for key, value in candidate.items():
        if key != "identifier":
            payload[key] = value
    return payload


# List outer test months in their fixed chronological order.
def outer_months(settings):
    months = []
    dates = pd.date_range(
        settings["first_outer_test_month"],
        settings["last_outer_test_month"],
        freq="MS",
    )
    for date in dates:
        months.append(date.strftime("%Y-%m"))
    return months


# Count unique inner panel fits rather than repeated requests from overlapping
# outer folds. One panel fit trains and evaluates all seventeen assets together.
def describe_plan(settings, candidates_by_arm):
    number_of_outer_months = len(outer_months(settings))
    unique_inner_months = number_of_outer_months + settings["inner_folds"] - 1

    candidate_count = 0
    for candidates in candidates_by_arm.values():
        candidate_count += len(candidates)

    inner_fits = candidate_count * unique_inner_months
    inner_fits *= len(settings["tuning_seeds"])
    outer_fits = len(settings["panel_arms"]) * number_of_outer_months
    outer_fits *= len(settings["evaluation_seeds"])
    plan = {
        "unique_inner_months": unique_inner_months,
        "inner_panel_fits": inner_fits,
        "outer_panel_fits": outer_fits,
        "total_panel_fits": inner_fits + outer_fits,
    }

    print("Relational nested walk-forward cross-validation")
    print(f"Cryptocurrencies in each panel: {len(settings['symbols'])}")
    print(f"Outer months: {number_of_outer_months}")
    print(f"Inner folds per outer month: {settings['inner_folds']}")
    print(f"Estimated panel fits: {plan['total_panel_fits']}")
    return plan


# Load datasets and chronological splits once, then keep graphs, features and
# repeated inner scores in memory for the duration of one execution.
def create_repository(settings):
    all_datasets = load_all_cryptocurrencies(settings["dataset_directory"])
    datasets = {}
    for symbol in settings["symbols"]:
        datasets[symbol] = all_datasets[symbol]

    reference = datasets[settings["symbols"][0]]
    nested = make_nested_monthly_splits(
        reference.index,
        first_outer_test_month=settings["first_outer_test_month"],
        last_outer_test_month=settings["last_outer_test_month"],
        inner_folds=settings["inner_folds"],
        forecast_horizon_hours=settings["forecast_horizon_hours"],
        target_valid=reference["target_valid"].to_numpy(dtype=bool),
    )
    nested_by_month = {}
    for fold in nested:
        month = fold["outer"]["test_start"].strftime("%Y-%m")
        nested_by_month[month] = fold

    return {
        "settings": settings,
        "datasets": datasets,
        "nested": nested_by_month,
        "features": {},
        "prepared_folds": {},
        "inner_results": {},
    }


# Keep the required graph context plus one earlier hour used to calculate the
# first causal log-ratio without inventing a previous Close.
def slice_datasets(repository, split):
    context_start = split["train_start"] - pd.DateOffset(hours=3)
    sliced = {}
    for symbol, frame in repository["datasets"].items():
        sliced[symbol] = frame.loc[
            (frame.index >= context_start) & (frame.index < split["test_end"])
        ]
    return sliced


# Load the partition whose cutoff equals the month being predicted. Its DTW
# window therefore contains only information available before that fold.
def load_cluster_assignments(repository, split, candidate):
    settings = repository["settings"]
    month = split["test_start"].strftime("%Y-%m")
    path = (
        settings["clustering_directory"]
        / f"cutoff_{month}"
        / f"window_{candidate['clustering_lookback_months']}m"
        / f"k_{candidate['number_of_clusters']}.json"
    )
    partition = json.loads(path.read_text(encoding="utf-8"))
    assignments = {}
    for symbol, cluster in partition["assignments"].items():
        assignments[symbol] = int(cluster)

    return assignments


# Prepare each distinct month and topology once. Neural candidates that differ
# only in optimizer or layer sizes reuse the same graph and standardized features.
def repository_prepared_fold(repository, split, candidate):
    month = split["test_start"].strftime("%Y-%m")
    key = (
        month,
        candidate["relation_mode"],
        candidate["clustering_lookback_months"],
        candidate["number_of_clusters"],
    )
    if key in repository["prepared_folds"]:
        return repository["prepared_folds"][key]

    datasets = slice_datasets(repository, split)
    assignments = None
    if candidate["relation_mode"] == "clustered":
        assignments = load_cluster_assignments(
            repository,
            split,
            candidate,
        )

    graph = build_panel_graph(
        datasets,
        candidate["relation_mode"],
        assignments,
    )

    # Topologies for the same month have identical nodes, so one standardized
    # feature matrix can be shared without sharing any learned model weights.
    if month not in repository["features"]:
        repository["features"][month] = build_standardized_panel_features(
            datasets,
            graph,
            split,
            repository["settings"]["feature_set"],
            repository["settings"]["forecast_horizon_hours"],
        )
    features = repository["features"][month]
    data = prepare_panel_fold_data(
        graph,
        features["values"],
        split,
        message_passing_layers=2,
        forecast_horizon_hours=repository["settings"]["forecast_horizon_hours"],
    )

    repository["prepared_folds"][key] = data
    return data


# Create the compact progress state used by the full 2,340-fit execution.
def create_progress(experiment_root, total_runs, report_every):
    progress = {
        "total_runs": total_runs,
        "completed_runs": 0,
        "report_every": report_every,
        "current_run": None,
    }
    save_progress(experiment_root, progress, "running")
    return progress


# Save only counts and the active fit
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


# Count only newly trained unique fits. Reused inner scores do not increase the
# counter because they are already included once in the 2,340-fit plan.
def complete_progress(experiment_root, progress):
    progress["completed_runs"] += 1
    progress["current_run"] = None
    save_progress(experiment_root, progress, "running")

    completed = progress["completed_runs"]
    if completed % progress["report_every"] == 0:
        percent = 100.0 * completed / progress["total_runs"]
        print(
            f"Progress: {completed}/{progress['total_runs']} ({percent:.2f}%)",
            flush=True,
        )


# Train one unique panel fold or return its in-memory inner score when an adjacent
# outer month requests the exact same candidate, month and seed.
def execute_run(
    repository,
    experiment_root,
    role,
    split,
    candidate,
    seed,
    progress,
):
    if role == "inner_selection":
        key = (split["test_start"], candidate["identifier"], seed)
        if key in repository["inner_results"]:
            return repository["inner_results"][key]

    progress["current_run"] = {
        "role": role,
        "arm": candidate["experimental_arm"],
        "month": split["test_start"].strftime("%Y-%m"),
        "candidate": candidate["identifier"],
        "seed": seed,
    }
    save_progress(experiment_root, progress, "running")

    data = repository_prepared_fold(repository, split, candidate)
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
    model, result = train_panel_gat(
        data,
        training_config,
        settings["symbols"],
    )
    del model

    if role == "inner_selection":
        payload = {"test_macro_f1": result["test_mean_asset_macro_f1"]}
        repository["inner_results"][key] = payload
    else:
        payload = result
    complete_progress(experiment_root, progress)
    return payload


# Flatten one asset's confusion matrix into readable columns. These counts are
# sufficient to derive all class metrics later without saving large reports.
def outer_symbol_columns(result, symbol):
    symbol_result = result["test_by_symbol"][symbol]
    columns = {
        "validation_mean_asset_macro_f1": result[
            "validation_mean_asset_macro_f1"
        ],
        "test_macro_f1": symbol_result["macro_f1"],
        "test_accuracy": symbol_result["accuracy"],
        "test_nodes": symbol_result["test_nodes"],
        "test_pooled_macro_f1": result["test_pooled_macro_f1"],
        "best_epoch": result["best_epoch"],
        "epochs_trained": result["epochs_trained"],
    }

    class_names = ("normal", "up", "down")
    matrix = symbol_result["confusion_matrix"]
    for actual_index, actual_name in enumerate(class_names):
        for predicted_index, predicted_name in enumerate(class_names):
            name = f"actual_{actual_name}_predicted_{predicted_name}"
            columns[name] = matrix[actual_index][predicted_index]
    return columns


# Choose the highest mean inner score, followed by lower variability and the
# readable candidate identifier as deterministic tie-breakers.
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


# Run candidate selection and outer evaluation without opening any outer test month
# before its model has been selected on earlier inner folds.
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
        nested = repository["nested"][outer_month]

        for arm in settings["panel_arms"]:
            scored_candidates = []
            for candidate in candidates_by_arm[arm]:
                scores = []
                for inner_split in nested["inner"]:
                    for seed in settings["tuning_seeds"]:
                        result = execute_run(
                            repository,
                            experiment_root,
                            "inner_selection",
                            inner_split,
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
                    "candidate_id": candidate["identifier"],
                    "mean_inner_asset_macro_f1": mean_score,
                    "std_inner_asset_macro_f1": score_std,
                    "inner_evaluations": len(scores),
                }
                row.update(candidate_payload(candidate))
                selection_rows.append(row)
                print(
                    f"Mean inner macro F1 | {outer_month} | {arm} | "
                    f"{candidate['identifier']}: {mean_score:.4f}",
                    flush=True,
                )

            selected = select_best_candidate(scored_candidates)
            print(
                f"Selected {outer_month} {arm}: {selected['identifier']}",
                flush=True,
            )

            for seed in settings["evaluation_seeds"]:
                result = execute_run(
                    repository,
                    experiment_root,
                    "outer_evaluation",
                    nested["outer"],
                    selected,
                    seed,
                    progress,
                )
                for symbol in settings["symbols"]:
                    row = {
                        "outer_month": outer_month,
                        "experimental_arm": arm,
                        "feature_set": settings["feature_set"],
                        "symbol": symbol,
                        "seed": seed,
                        "selected_candidate_id": selected["identifier"],
                    }
                    row.update(candidate_payload(selected))
                    row.update(outer_symbol_columns(result, symbol))
                    outer_rows.append(row)

            write_csv(
                experiment_root / "inner_model_selection.csv",
                pd.DataFrame(selection_rows),
            )
            write_csv(
                experiment_root / "outer_results.csv",
                pd.DataFrame(outer_rows),
            )


# Build candidates, record the complete plan and run the selected configuration.
def run_nested_cv(settings):
    candidates_by_arm = {}
    for arm in settings["panel_arms"]:
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
        plan["total_panel_fits"],
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


# Run the complete relational experiment defined in the focused configuration.
if __name__ == "__main__":
    run_nested_cv(EXPERIMENT)
