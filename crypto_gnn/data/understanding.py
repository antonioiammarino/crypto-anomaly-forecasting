# Description and quality checks for the raw cryptocurrency data.

from pathlib import Path

import numpy as np
import pandas as pd


MARKET_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = PROJECT_ROOT / "dataset crypto" / "cryptos_merged"
DESCRIPTION_OUTPUT = PROJECT_ROOT / "artifacts" / "data_description.csv"
QUALITY_OUTPUT = PROJECT_ROOT / "artifacts" / "data_quality.csv"
DEVELOPMENT_END = pd.Timestamp("2022-01-01", tz="UTC")


# Read the raw files without applying targets or model transformations. Data
# Understanding must describe the source data before Data Preparation changes it.
def load_raw_datasets(dataset_directory):
    datasets = {}
    for path in sorted(Path(dataset_directory).glob("*.csv")):
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame["Date"])

        # Local timestamps are converted to UTC only to select the exact development
        # period. The repeated autumn hour is valid and becomes unique in UTC.
        local_dates = dates.dt.tz_localize("Europe/Rome", ambiguous="infer")
        frame["timestamp_utc"] = local_dates.dt.tz_convert("UTC")
        datasets[path.stem] = frame
    return datasets


# Calculate descriptive statistics on the development period. The confirmatory
# year is excluded because it was kept aside until the protocol had been frozen.
def describe_data(datasets, development_end=DEVELOPMENT_END):
    rows = []
    for symbol, frame in sorted(datasets.items()):
        development = frame.loc[frame["timestamp_utc"] < development_end]
        for feature in MARKET_COLUMNS:
            values = pd.to_numeric(development[feature], errors="coerce")
            observed = values.dropna()
            rows.append(
                {
                    "symbol": symbol,
                    "feature": feature,
                    "rows": len(values),
                    "observed": len(observed),
                    "missing": int(values.isna().sum()),
                    "missing_pct": 100.0 * values.isna().mean(),
                    "unique_values": int(observed.nunique()),
                    "minimum": observed.min(),
                    "maximum": observed.max(),
                    "mean": observed.mean(),
                    "median": observed.median(),
                    "standard_deviation": observed.std(),
                    "zero_count": int(observed.eq(0).sum()),
                    "zero_pct": 100.0 * observed.eq(0).mean(),
                }
            )
    return pd.DataFrame(rows)


# Check completeness, timestamp consistency and simple domain constraints. The
# report uses the same development period as the descriptive statistics so no
# property of the confirmatory year can influence experimental decisions.
def verify_data_quality(datasets, development_end=DEVELOPMENT_END):
    rows = []
    for symbol, frame in sorted(datasets.items()):
        development = frame.loc[frame["timestamp_utc"] < development_end].copy()
        numeric = development.loc[:, MARKET_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        missing = numeric.isna()
        timestamps = pd.DatetimeIndex(development["timestamp_utc"])
        hourly_nanoseconds = 60 * 60 * 1_000_000_000
        irregular_steps = np.sum(np.diff(timestamps.asi8) != hourly_nanoseconds)
        adjacent_close_matches = np.isclose(
            numeric["Close"],
            numeric["Adj Close"],
            equal_nan=True,
        )

        # Rows with text in a numeric column are different from genuinely empty
        # cells, so they are counted separately as a consistency problem.
        original_market = development.loc[:, MARKET_COLUMNS]
        non_numeric = numeric.isna() & original_market.notna()
        volume_observed = numeric["Volume"].notna()
        zero_volume = numeric["Volume"].eq(0) & volume_observed

        rows.append(
            {
                "symbol": symbol,
                "rows": len(development),
                "start_utc": timestamps.min(),
                "end_utc": timestamps.max(),
                "missing_cells": int(missing.sum().sum()),
                "rows_with_missing_values": int(missing.any(axis=1).sum()),
                "rows_all_market_values_missing": int(missing.all(axis=1).sum()),
                "rows_partly_missing": int(
                    (missing.any(axis=1) & ~missing.all(axis=1)).sum()
                ),
                "non_numeric_cells": int(non_numeric.sum().sum()),
                "duplicate_local_timestamp_rows": int(
                    development["Date"].duplicated(keep=False).sum()
                ),
                "duplicate_utc_timestamp_rows": int(timestamps.duplicated().sum()),
                "irregular_utc_steps": int(irregular_steps),
                "out_of_order_utc_rows": int(
                    np.sum(timestamps[1:].asi8 < timestamps[:-1].asi8)
                ),
                "negative_price_cells": int(
                    numeric.loc[:, ["Open", "High", "Low", "Close"]]
                    .lt(0)
                    .sum()
                    .sum()
                ),
                "negative_volume_rows": int(numeric["Volume"].lt(0).sum()),
                "zero_volume_rows": int(zero_volume.sum()),
                "zero_volume_pct": 100.0 * zero_volume.sum() / volume_observed.sum(),
                "adjusted_close_mismatches": int((~adjacent_close_matches).sum()),
            }
        )
    return pd.DataFrame(rows)


# Generate the two Data Understanding reports. They are descriptive outputs and
# are never read by training, which continues to prepare every fold in memory.
if __name__ == "__main__":
    datasets = load_raw_datasets(DATASET_DIRECTORY)
    description = describe_data(datasets)
    quality = verify_data_quality(datasets)

    DESCRIPTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    description.to_csv(DESCRIPTION_OUTPUT, index=False)
    quality.to_csv(QUALITY_OUTPUT, index=False)

    display_columns = [
        "symbol",
        "rows",
        "rows_with_missing_values",
        "rows_partly_missing",
        "zero_volume_pct",
        "duplicate_utc_timestamp_rows",
        "irregular_utc_steps",
    ]
    display = quality.loc[:, display_columns].copy()
    display["zero_volume_pct"] = display["zero_volume_pct"].round(2)
    print(display.to_string(index=False))
    print(f"\nDescription written to: {DESCRIPTION_OUTPUT}")
    print(f"Quality report written to: {QUALITY_OUTPUT}")
