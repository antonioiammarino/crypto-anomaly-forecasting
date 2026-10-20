# Loading and causal preprocessing of hourly cryptocurrency data.

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

LABEL_INVALID = -1
LABEL_NORMAL = 0
LABEL_UP = 1
LABEL_DOWN = 2
LABEL_NAMES = {
    LABEL_INVALID: "invalid",
    LABEL_NORMAL: "normal",
    LABEL_UP: "up",
    LABEL_DOWN: "down",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = PROJECT_ROOT / "dataset crypto" / "cryptos_merged"
SUMMARY_OUTPUT = PROJECT_ROOT / "artifacts" / "data_quality_summary.csv"
PROCESSED_DIRECTORY = None
ANOMALY_THRESHOLD = 0.01


# Convert one raw cryptocurrency series to a common UTC timeline and create the
# one-hour-ahead target used by every experiment.
def preprocess_frame(
    frame,
    symbol,
    anomaly_threshold=0.01,
    source_timezone="Europe/Rome",
):
    raw = frame.copy()

    # The CSV stores Italian local time. UTC gives every asset one unambiguous
    # timeline, including the repeated hour when daylight saving time ends.
    dates = pd.to_datetime(raw["Date"])
    localized_dates = dates.dt.tz_localize(source_timezone, ambiguous="infer")
    utc_index = pd.DatetimeIndex(
        localized_dates.dt.tz_convert("UTC"),
        name="timestamp",
    )
    raw.index = utc_index

    # Missing market observations remain missing. They are marked instead of
    # being imputed because invented prices would change graphs and labels.
    numeric_raw = raw.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    raw_feature_valid = numeric_raw.notna().all(axis=1)
    raw_close_valid = numeric_raw["Close"].notna()

    result = numeric_raw.astype("float64").copy()
    result.insert(0, "symbol", symbol)
    result["observation_valid"] = raw_feature_valid

    # The future price is used only to create the label. It is never included in
    # model features, which keeps prediction causal.
    future_close = numeric_raw["Close"].shift(-1)
    future_return = future_close.div(numeric_raw["Close"]).sub(1.0)
    target_valid = raw_close_valid & raw_close_valid.shift(-1, fill_value=False)

    # Invalid targets use -1
    # Valid returns inside the two anomaly thresholds belong to the normal class.
    labels = pd.Series(LABEL_INVALID, index=result.index, dtype="int8")
    labels.loc[target_valid] = LABEL_NORMAL
    labels.loc[target_valid & future_return.ge(anomaly_threshold)] = LABEL_UP
    labels.loc[target_valid & future_return.le(-anomaly_threshold)] = LABEL_DOWN

    result["future_return"] = future_return.astype("float64")
    result["target"] = labels
    result["target_name"] = labels.map(LABEL_NAMES)
    result["target_valid"] = target_valid.astype(bool)

    return result


# Load one CSV and use its filename as the cryptocurrency symbol. This keeps the
# dataset convention in one place instead of repeating it in later experiments.
def load_cryptocurrency(
    path,
    anomaly_threshold=0.01,
    source_timezone="Europe/Rome",
):
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    return preprocess_frame(
        frame,
        symbol=csv_path.stem,
        anomaly_threshold=anomaly_threshold,
        source_timezone=source_timezone,
    )


# Load every cryptocurrency in alphabetical order. The fixed order makes later
# multi-asset experiments reproducible.
def load_all_cryptocurrencies(
    dataset_directory,
    anomaly_threshold=0.01,
    source_timezone="Europe/Rome",
):
    directory = Path(dataset_directory)
    paths = sorted(directory.glob("*.csv"))

    datasets = {}
    for path in paths:
        datasets[path.stem] = load_cryptocurrency(
            path,
            anomaly_threshold=anomaly_threshold,
            source_timezone=source_timezone,
        )
    return datasets


# Summarize missing observations and class balance. Percentages use only valid
# targets because rows next to missing prices do not have a real future return.
def summarize_dataset(datasets):
    rows = []
    for symbol, frame in sorted(datasets.items()):
        valid = frame.loc[frame["target_valid"]]
        counts = valid["target"].value_counts()
        valid_count = len(valid)

        rows.append(
            {
                "symbol": symbol,
                "rows": len(frame),
                "start_utc": frame.index.min(),
                "end_utc": frame.index.max(),
                "missing_rows": int((~frame["observation_valid"]).sum()),
                "valid_targets": valid_count,
                "normal": int(counts.get(LABEL_NORMAL, 0)),
                "up": int(counts.get(LABEL_UP, 0)),
                "down": int(counts.get(LABEL_DOWN, 0)),
                "normal_pct": 100.0 * counts.get(LABEL_NORMAL, 0) / valid_count,
                "up_pct": 100.0 * counts.get(LABEL_UP, 0) / valid_count,
                "down_pct": 100.0 * counts.get(LABEL_DOWN, 0) / valid_count,
                "zero_volume_pct": 100.0 * frame["Volume"].eq(0).mean(),
            }
        )

    return pd.DataFrame(rows)


# Prepare all cryptocurrencies and save the summary used to inspect the dataset
# before starting expensive graph construction and training.
if __name__ == "__main__":
    datasets = load_all_cryptocurrencies(
        DATASET_DIRECTORY,
        anomaly_threshold=ANOMALY_THRESHOLD,
    )
    summary = summarize_dataset(datasets)

    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    if PROCESSED_DIRECTORY is not None:
        PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
        for symbol, frame in datasets.items():
            output_path = PROCESSED_DIRECTORY / f"{symbol}.csv"
            frame.to_csv(output_path, index_label="timestamp")

    display = summary.copy()
    percentage_columns = [
        "normal_pct",
        "up_pct",
        "down_pct",
        "zero_volume_pct",
    ]
    display[percentage_columns] = display[percentage_columns].round(2)
    print(display.to_string(index=False))
    print(f"\nSummary written to: {SUMMARY_OUTPUT}")

