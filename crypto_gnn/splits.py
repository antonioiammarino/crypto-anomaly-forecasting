import numpy as np
import pandas as pd


# Normalize any date to the first day of its month in UTC. The first day is used
# because every train, validation and test interval follows calendar months.
def _as_utc_month_start(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return pd.Timestamp(
        year=timestamp.year,
        month=timestamp.month,
        day=1,
        tz="UTC",
    )


# Create monthly rolling folds in chronological order. Each test fold represents
# one complete month, while all training and validation observations come before it.
def make_monthly_rolling_splits(
    index,
    first_test_month="2020-07-01",
    last_test_month="2021-12-01",
    history_months=6,
    validation_months=1,
    forecast_horizon_hours=1,
    target_valid=None,
    require_full_history=True,
):
    first_test = _as_utc_month_start(first_test_month)
    last_test = _as_utc_month_start(last_test_month)

    # Invalid targets come from missing prices and must not enter any partition.
    valid_mask = np.ones(len(index), dtype=bool)
    if target_valid is not None:
        valid_mask = np.asarray(target_valid, dtype=bool)

    folds = []
    label_times = index + pd.Timedelta(hours=forecast_horizon_hours)
    month_starts = pd.date_range(first_test, last_test, freq="MS", tz="UTC")
    for test_start in month_starts:
        train_start = test_start - pd.DateOffset(months=history_months)
        validation_start = test_start - pd.DateOffset(months=validation_months)
        test_end = test_start + pd.DateOffset(months=1)

        # Skip incomplete folds because all experiments must use the same history.
        if require_full_history and index.min() > train_start:
            continue
        if index.max() < test_end - pd.DateOffset(hours=1):
            continue

        # A target at time t uses the price at t + horizon. The last observations
        # before each boundary are removed so their future price cannot leak forward.
        train_mask = (
            (index >= train_start)
            & (index < validation_start)
            & (label_times < validation_start)
            & valid_mask
        )
        validation_mask = (
            (index >= validation_start)
            & (index < test_start)
            & (label_times < test_start)
            & valid_mask
        )
        test_mask = (
            (index >= test_start)
            & (index < test_end)
            & (label_times < test_end)
            & valid_mask
        )

        split = {
            "fold": len(folds),
            "train_start": train_start,
            "validation_start": validation_start,
            "test_start": test_start,
            "test_end": test_end,
            "train_mask": np.asarray(train_mask, dtype=bool),
            "validation_mask": np.asarray(validation_mask, dtype=bool),
            "test_mask": np.asarray(test_mask, dtype=bool),
        }
        folds.append(split)

    return folds


# Create nested monthly folds for model selection. Inner folds end before the outer
# test month so hyperparameters never use the period reserved for final evaluation.
def make_nested_monthly_splits(
    index,
    first_outer_test_month="2021-01-01",
    last_outer_test_month="2021-12-01",
    inner_folds=5,
    history_months=6,
    validation_months=1,
    forecast_horizon_hours=1,
    target_valid=None,
):
    first_outer = _as_utc_month_start(first_outer_test_month)
    last_outer = _as_utc_month_start(last_outer_test_month)

    nested_folds = []
    outer_months = pd.date_range(first_outer, last_outer, freq="MS", tz="UTC")
    for outer_test_start in outer_months:
        outer = make_monthly_rolling_splits(
            index,
            first_test_month=outer_test_start,
            last_test_month=outer_test_start,
            history_months=history_months,
            validation_months=validation_months,
            forecast_horizon_hours=forecast_horizon_hours,
            target_valid=target_valid,
            require_full_history=True,
        )[0]

        # Select the requested number of months immediately before the outer test.
        last_inner_test = outer_test_start - pd.DateOffset(months=1)
        first_inner_test = last_inner_test - pd.DateOffset(months=inner_folds - 1)
        inner = make_monthly_rolling_splits(
            index,
            first_test_month=first_inner_test,
            last_test_month=last_inner_test,
            history_months=history_months,
            validation_months=validation_months,
            forecast_horizon_hours=forecast_horizon_hours,
            target_valid=target_valid,
            require_full_history=True,
        )
        nested_folds.append({"outer": outer, "inner": inner})

    return nested_folds

