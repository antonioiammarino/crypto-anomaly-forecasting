# Causal feature construction, temporal windows and train-only standardization.

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

OHLC_FEATURE_NAMES = (
    "log_open_to_previous_close",
    "log_high_to_previous_close",
    "log_low_to_previous_close",
    "log_close_to_previous_close",
)
MODEL_FEATURE_NAMES = OHLC_FEATURE_NAMES + ("log1p_volume",)
FEATURE_SET_NAMES = {
    "ohlc": OHLC_FEATURE_NAMES,
    "ohlcv": MODEL_FEATURE_NAMES,
}


# Concatenate current and past features for the temporal MLP. A missing hour breaks
# the window because observations on opposite sides of a gap are not consecutive.
def build_causal_feature_window(features, source_positions, window_hours):
    features = np.asarray(features, dtype=np.float32)
    source_positions = np.asarray(source_positions, dtype=np.int64)

    nodes, feature_count = features.shape
    feature_width = window_hours * feature_count
    windowed = np.zeros(
        (nodes, feature_width + window_hours - 1),
        dtype=np.float32,
    )

    # Lag zero is the current node. Older values are copied only when their source
    # positions prove that every required hourly observation exists.
    for lag in range(window_hours):
        target_nodes = np.arange(lag, nodes)
        source_nodes = target_nodes - lag
        available = (
            source_positions[target_nodes] - source_positions[source_nodes] == lag
        )
        valid_targets = target_nodes[available]
        valid_sources = source_nodes[available]
        block = window_hours - 1 - lag
        start = block * feature_count
        stop = start + feature_count
        windowed[valid_targets, start:stop] = features[valid_sources]
        if lag > 0:
            windowed[valid_targets, feature_width + block] = 1.0

    return windowed


# Return names in the exact order used by the temporal window. Keeping the same
# order makes saved results and feature inspection easier to interpret.
def temporal_window_feature_names(base_names, window_hours):
    names = []
    for lag in range(window_hours - 1, -1, -1):
        for name in base_names:
            names.append(f"lag_{lag}h__{name}")

    for lag in range(window_hours - 1, 0, -1):
        names.append(f"lag_{lag}h__available")

    return tuple(names)


# Convert raw OHLC or OHLCV values into log-ratios. Relative changes are comparable
# across cryptocurrencies even when their nominal prices are very different.
def build_causal_features(frame, feature_set="ohlcv"):
    if feature_set == "ohlcv":
        requested_columns = FEATURE_COLUMNS
    else:
        requested_columns = FEATURE_COLUMNS[:-1]

    values = frame.loc[:, requested_columns].astype("float64")
    prices = values.loc[:, ["Open", "High", "Low", "Close"]]

    # Use the previous Close only inside a continuous hourly segment. After a gap,
    # the current Close becomes the reference so no missing period is crossed.
    previous_close = values["Close"].shift(1)
    segment_start = np.zeros(len(values), dtype=bool)
    segment_start[0] = True
    differences = frame.index.to_series().diff().to_numpy(dtype="timedelta64[ns]")
    segment_start[1:] = differences[1:] != np.timedelta64(1, "h")
    previous_close.loc[segment_start] = values.loc[segment_start, "Close"]

    price_features = np.log(prices.div(previous_close, axis=0))
    columns = [
        price_features["Open"],
        price_features["High"],
        price_features["Low"],
        price_features["Close"],
    ]
    # log1p keeps zero volume valid and reduces the large scale differences.
    if feature_set == "ohlcv":
        columns.append(np.log1p(values["Volume"]))
    return np.column_stack(columns).astype(np.float32)


# Build the complete OHLCV feature set used by the default experiments.
def build_causal_ohlcv_features(frame):
    return build_causal_features(frame, feature_set="ohlcv")


# Estimate normalization statistics on training observations only. Validation and
# test values must not influence the scale seen during model fitting.
def fit_feature_standardizer(features, training_mask):
    features = np.asarray(features, dtype=np.float64)
    training_mask = np.asarray(training_mask, dtype=bool)
    training_features = features[training_mask]

    mean = training_features.mean(axis=0).astype(np.float32)
    # A constant feature uses scale 1 so standardization returns zero safely.
    scale = training_features.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    return {"mean": mean, "scale": scale}


# Apply the frozen training statistics to every partition. Reusing the same values
# prevents information from validation or test data entering preprocessing.
def standardize_features(features, standardizer):
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(standardizer["mean"], dtype=np.float32)
    scale = np.asarray(standardizer["scale"], dtype=np.float32)
    return ((features - mean) / scale).astype(np.float32)

