# Exact Dynamic Time Warping distances between aligned price trajectories.

import sys

import numpy as np


DTW_DEFINITION = "sqrt_minimum_sum_squared_differences"


# Calculate exact unconstrained DTW with two rows of working memory. This clear
# implementation is used for short tests, while the full experiment uses the
# equivalent compiled implementation.
def dtw_distance_reference(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    previous = np.full(len(second) + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0

    for first_value in first:
        current = np.full(len(second) + 1, np.inf, dtype=np.float64)
        for column, second_value in enumerate(second, start=1):
            cost = float((first_value - second_value) ** 2)
            current[column] = cost + min(
                current[column - 1],
                previous[column],
                previous[column - 1],
            )
        previous = current

    return float(np.sqrt(previous[-1]))


# Compute every pair once and mirror it because DTW is symmetric here. The
# compiled backend stays in this separate workflow because its OpenMP runtime can
# conflict with PyTorch on macOS.
def pairwise_dtw_distance_matrix(values, backend="dtaidistance"):
    series = np.asarray(values, dtype=np.float64)

    if backend == "reference":
        matrix = np.zeros((len(series), len(series)), dtype=np.float64)
        for left in range(len(series)):
            for right in range(left + 1, len(series)):
                distance = dtw_distance_reference(series[left], series[right])
                matrix[left, right] = distance
                matrix[right, left] = distance
    else:
        if "torch" in sys.modules:
            raise RuntimeError(
                "Run clustering in a separate process before importing PyTorch."
            )

        from dtaidistance import dtw

        contiguous_series = []
        for row in series:
            contiguous_series.append(np.ascontiguousarray(row))

        matrix = np.asarray(
            dtw.distance_matrix_fast(
                contiguous_series,
                compact=False,
                parallel=False,
                use_pruning=True,
            ),
            dtype=np.float64,
        )

    np.fill_diagonal(matrix, 0.0)
    return matrix
