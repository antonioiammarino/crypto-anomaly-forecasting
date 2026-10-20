# Frozen walk-forward evaluation over the twelve months of 2022.

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_gnn.data.preparation.features import (
    build_causal_feature_window,
    fit_feature_standardizer,
    standardize_features,
)
from crypto_gnn.training.common import (
    evaluate_majority_baseline,
    prepare_fold_data,
    train_neural_baseline,
)
from crypto_gnn.training.config_confirmatory import (
    CONFIRMATION,
    CONFIRMATORY_PROTOCOL_NAME,
    FEATURE_SETS,
    INDEPENDENT_ARMS,
    PANEL_ARMS,
)
from crypto_gnn.training.config_independent import EXPERIMENT as INDEPENDENT_EXPERIMENT
from crypto_gnn.training.config_relational import EXPERIMENT as RELATIONAL_EXPERIMENT
from crypto_gnn.training.independent import (
    create_repository as create_independent_repository,
)
from crypto_gnn.training.independent import (
    outer_result_columns,
    repository_features,
    repository_graph,
    repository_splits,
)
from crypto_gnn.training.panel import train_panel_gat
from crypto_gnn.training.protocol import load_frozen_protocol
from crypto_gnn.training.relational import (
    candidate_payload,
    create_repository as create_panel_repository,
)
from crypto_gnn.training.relational import (
    outer_symbol_columns,
    repository_prepared_fold,
)
from crypto_gnn.training.search import MAJORITY_CANDIDATE


# Convert paths when the complete frozen settings are recorded as JSON.
def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


# Save configuration and progress safely during the long full experiment.
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


# Store one seed-level outer table. All averages and statistical comparisons are
# derived later by the evaluation workflow.
def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


# List the twelve confirmatory test months in chronological order.
def confirmatory_months():
    months = []
    dates = pd.date_range(
        CONFIRMATION["first_test_month"],
        CONFIRMATION["last_test_month"],
        freq="MS",
    )
    for date in dates:
        months.append(date.strftime("%Y-%m"))
    return months


# Count actual model fits. One panel fit evaluates all seventeen assets together,
# whereas each independent model is fitted separately for every asset.
def describe_plan():
    months = len(confirmatory_months())
    symbols = len(CONFIRMATION["symbols"])
    seeds = len(CONFIRMATION["evaluation_seeds"])
    majority = months * symbols
    independent = months * len(INDEPENDENT_ARMS) * len(FEATURE_SETS)
    independent *= symbols * seeds
    panel = months * len(PANEL_ARMS) * seeds
    return {
        "majority_evaluations": majority,
        "independent_neural_fits": independent,
        "panel_neural_fits": panel,
        "total_executions": majority + independent + panel,
    }


# Adapt the development repositories to 2022 without reopening either search
# space. Each monthly split still uses five training months and one validation month.
def repository_settings(frozen):
    confirmation = frozen["confirmation"]
    independent = dict(INDEPENDENT_EXPERIMENT)
    independent.update(
        {
            "first_outer_test_month": confirmation["first_test_month"],
            "last_outer_test_month": confirmation["last_test_month"],
            "evaluation_seeds": tuple(confirmation["evaluation_seeds"]),
            "maximum_epochs": confirmation["maximum_epochs"],
            "patience": confirmation["patience"],
            "device": confirmation["device"],
        }
    )

    panel = dict(RELATIONAL_EXPERIMENT)
    panel.update(
        {
            "first_outer_test_month": confirmation["first_test_month"],
            "last_outer_test_month": confirmation["last_test_month"],
            "evaluation_seeds": tuple(confirmation["evaluation_seeds"]),
            "maximum_epochs": confirmation["maximum_epochs"],
            "patience": confirmation["patience"],
            "device": confirmation["device"],
            "clustering_directory": (
                CONFIRMATION["frozen_protocol_directory"] / "clusters"
            ),
        }
    )
    return independent, panel


# Fit one frozen independent candidate on one asset, month and random seed.
def evaluate_independent(repository, month, symbol, feature_set, candidate, seed):
    split = repository_splits(repository, symbol)[month]["outer"]
    graph = repository_graph(repository, symbol, candidate)
    graph_training_mask = split["train_mask"][graph["source_positions"]]

    if candidate["architecture"] == "majority":
        features = np.zeros((graph["number_of_nodes"], 1), dtype=np.float32)
        data = prepare_fold_data(graph, features, split, edge_mode="temporal")
        return evaluate_majority_baseline(data)

    raw_features = repository_features(repository, symbol, feature_set, graph)
    standardizer = fit_feature_standardizer(raw_features, graph_training_mask)
    features = standardize_features(raw_features, standardizer)
    if candidate["architecture"] == "mlp":
        features = build_causal_feature_window(
            features,
            graph["source_positions"],
            candidate["temporal_window_hours"],
        )

    data = prepare_fold_data(
        graph,
        features,
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
    model, result = train_neural_baseline(
        data,
        training_config,
        candidate["architecture"],
    )
    del model
    return result


# Fit one frozen panel candidate. Its result contains separate metrics for all
# assets, so the outer table is expanded only after this shared fit is complete.
def evaluate_panel(repository, month, candidate, seed):
    split = repository["nested"][month]["outer"]
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
    model, result = train_panel_gat(data, training_config, settings["symbols"])
    del model
    return result


# Save a small progress file and print only the fit count and percentage.
def update_progress(output_directory, progress, status="running"):
    write_json(
        output_directory / "progress.json",
        {
            "protocol": CONFIRMATORY_PROTOCOL_NAME,
            "status": status,
            "total_runs": progress["total_runs"],
            "completed_runs": progress["completed_runs"],
            "current_run": progress["current_run"],
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
    completed = progress["completed_runs"]
    if (
        status == "running"
        and completed > 0
        and completed % progress["report_every"] == 0
    ):
        percent = 100.0 * completed / progress["total_runs"]
        print(f"Progress: {completed}/{progress['total_runs']} ({percent:.2f}%)")


# Run the full frozen confirmation. There is no candidate search and no short mode
# because development checks are kept outside this public workflow.
if __name__ == "__main__":
    frozen = load_frozen_protocol()
    independent_settings, panel_settings = repository_settings(frozen)
    independent_repository = create_independent_repository(independent_settings)
    panel_repository = create_panel_repository(panel_settings)
    selected = frozen["selected_candidates"]
    plan = describe_plan()
    output_directory = CONFIRMATION["output_directory"]

    write_json(
        output_directory / "experiment_config.json",
        {
            "protocol": CONFIRMATORY_PROTOCOL_NAME,
            "frozen_protocol": frozen["protocol"],
            "no_hyperparameter_search": True,
            "months": confirmatory_months(),
            "evaluation_seeds": list(CONFIRMATION["evaluation_seeds"]),
            "selected_candidates": {
                key: value["candidate"] for key, value in selected.items()
            },
            "plan": plan,
        },
    )

    print("Frozen confirmatory walk-forward evaluation")
    print(f"Test months: {len(confirmatory_months())}")
    print(f"Cryptocurrencies: {len(CONFIRMATION['symbols'])}")
    print("Hyperparameter search: none")
    print(f"Estimated model fits: {plan['total_executions']}")

    progress = {
        "total_runs": plan["total_executions"],
        "completed_runs": 0,
        "current_run": None,
        "report_every": CONFIRMATION["progress_report_every"],
    }
    update_progress(output_directory, progress)
    rows = []

    try:
        for month in confirmatory_months():
            print(f"Starting confirmatory month {month}")

            for symbol in CONFIRMATION["symbols"]:
                progress["current_run"] = {
                    "month": month,
                    "arm": "majority",
                    "symbol": symbol,
                    "seed": 0,
                }
                result = evaluate_independent(
                    independent_repository,
                    month,
                    symbol,
                    "none",
                    MAJORITY_CANDIDATE,
                    0,
                )
                row = {
                    "outer_month": month,
                    "experimental_arm": "majority",
                    "feature_set": "none",
                    "symbol": symbol,
                    "seed": 0,
                    "selected_candidate_id": MAJORITY_CANDIDATE["identifier"],
                }
                row.update(outer_result_columns(result))
                rows.append(row)
                progress["completed_runs"] += 1
                progress["current_run"] = None
                update_progress(output_directory, progress)

            for arm in INDEPENDENT_ARMS:
                for feature_set in FEATURE_SETS:
                    key = f"{arm}__{feature_set}"
                    candidate = selected[key]["candidate"]
                    for symbol in CONFIRMATION["symbols"]:
                        for seed in CONFIRMATION["evaluation_seeds"]:
                            progress["current_run"] = {
                                "month": month,
                                "arm": arm,
                                "feature_set": feature_set,
                                "symbol": symbol,
                                "seed": seed,
                            }
                            result = evaluate_independent(
                                independent_repository,
                                month,
                                symbol,
                                feature_set,
                                candidate,
                                seed,
                            )
                            row = {
                                "outer_month": month,
                                "experimental_arm": arm,
                                "feature_set": feature_set,
                                "symbol": symbol,
                                "seed": seed,
                                "selected_candidate_id": candidate["identifier"],
                            }
                            row.update(outer_result_columns(result))
                            rows.append(row)
                            progress["completed_runs"] += 1
                            progress["current_run"] = None
                            update_progress(output_directory, progress)

            for arm in PANEL_ARMS:
                candidate = selected[arm]["candidate"]
                for seed in CONFIRMATION["evaluation_seeds"]:
                    progress["current_run"] = {
                        "month": month,
                        "arm": arm,
                        "symbol": "panel",
                        "seed": seed,
                    }
                    result = evaluate_panel(
                        panel_repository,
                        month,
                        candidate,
                        seed,
                    )
                    for symbol in CONFIRMATION["symbols"]:
                        row = {
                            "outer_month": month,
                            "experimental_arm": arm,
                            "feature_set": "ohlc",
                            "symbol": symbol,
                            "seed": seed,
                            "selected_candidate_id": candidate["identifier"],
                        }
                        row.update(candidate_payload(candidate))
                        row.update(outer_symbol_columns(result, symbol))
                        rows.append(row)
                    progress["completed_runs"] += 1
                    progress["current_run"] = None
                    update_progress(output_directory, progress)

            write_csv(output_directory / "outer_results.csv", pd.DataFrame(rows))

    except BaseException:
        update_progress(output_directory, progress, status="failed")
        raise

    update_progress(output_directory, progress, status="completed")
    print(f"Confirmatory experiment completed: {output_directory}")
