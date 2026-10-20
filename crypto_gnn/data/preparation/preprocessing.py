# Loading and causal preparation of hourly cryptocurrency data.

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORY = PROJECT_ROOT / "dataset crypto" / "cryptos_merged"
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

    # Invalid targets use -1 so they cannot be confused with the normal class.
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



