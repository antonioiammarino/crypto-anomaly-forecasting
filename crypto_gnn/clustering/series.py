# Construction of causal price trajectories for DTW clustering.

import numpy as np
import pandas as pd


CLUSTERING_TRANSFORMATION = "per_symbol_zscore_log_close_ddof0"


# Build one aligned trajectory for every asset using only the months before the
# cutoff. Incomplete shared hours are removed because imputing prices would change
# the shapes compared by DTW.
def build_clustering_series(datasets, cutoff, lookback_months):
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    window_start = cutoff - pd.DateOffset(months=lookback_months)
    symbols = tuple(sorted(datasets))
    index = datasets[symbols[0]].index
    window_mask = (index >= window_start) & (index < cutoff)
    candidate_hours = int(window_mask.sum())

    # All columns use the same timeline. A value is retained only when the raw
    # observation is real for that asset, so no missing market hour is invented.
    close_columns = {}
    for symbol in symbols:
        frame = datasets[symbol]
        close = frame.loc[window_mask, "Close"].astype("float64").copy()
        observed = frame.loc[window_mask, "observation_valid"].to_numpy(dtype=bool)
        close.loc[~observed] = np.nan
        close_columns[symbol] = close

    shared_close = pd.DataFrame(close_columns).dropna(axis=0, how="any")

    # Log prices turn multiplicative price changes into additive movements. Each
    # asset is standardized separately so DTW compares trajectory shape rather
    # than nominal price level or scale.
    log_close = np.log(shared_close.to_numpy(dtype=np.float64))
    means = log_close.mean(axis=0)
    scales = log_close.std(axis=0, ddof=0)
    standardized = (log_close - means) / scales

    return {
        "symbols": symbols,
        "timestamps": pd.DatetimeIndex(shared_close.index.copy()),
        "values": np.ascontiguousarray(standardized.T, dtype=np.float64),
        "cutoff": cutoff,
        "window_start": window_start,
        "lookback_months": lookback_months,
        "candidate_hours": candidate_hours,
        "complete_hours": len(shared_close),
    }


# Keep the causal window details next to every saved distance matrix. These values
# make it possible to confirm how many incomplete hours were excluded.
def clustering_series_metadata(series):
    return {
        "symbols": list(series["symbols"]),
        "cutoff": series["cutoff"].isoformat(),
        "window_start": series["window_start"].isoformat(),
        "lookback_months": series["lookback_months"],
        "candidate_hours": series["candidate_hours"],
        "complete_hours": series["complete_hours"],
        "discarded_incomplete_hours": (
            series["candidate_hours"] - series["complete_hours"]
        ),
        "first_complete_timestamp": series["timestamps"].min().isoformat(),
        "last_complete_timestamp": series["timestamps"].max().isoformat(),
        "transformation": CLUSTERING_TRANSFORMATION,
    }
