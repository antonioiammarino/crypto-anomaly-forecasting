# Deterministic PAM k-medoids applied to a precomputed distance matrix.

import numpy as np


# Assign every asset to its nearest medoid and return the total distance. The
# objective is reused during both the BUILD and SWAP stages of PAM.
def assign_to_medoids(distance_matrix, medoids):
    distances = distance_matrix[:, medoids]
    labels = np.argmin(distances, axis=1).astype(np.int64)
    rows = np.arange(len(distance_matrix))
    objective = float(distances[rows, labels].sum())
    return labels, objective


# Calculate silhouette directly from the DTW matrix. Singleton clusters receive
# zero because their within-cluster distance cannot be estimated.
def silhouette_from_distance_matrix(distance_matrix, labels):
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    clusters = np.unique(labels)
    scores = np.zeros(len(matrix), dtype=np.float64)

    for item in range(len(matrix)):
        own_members = np.flatnonzero(labels == labels[item])
        own_members = own_members[own_members != item]
        if len(own_members) == 0:
            continue

        within = float(matrix[item, own_members].mean())
        nearest_other = np.inf
        for cluster in clusters:
            if cluster == labels[item]:
                continue
            other_members = labels == cluster
            distance = float(matrix[item, other_members].mean())
            nearest_other = min(nearest_other, distance)

        denominator = max(within, nearest_other)
        if denominator > 0:
            scores[item] = (nearest_other - within) / denominator

    return float(scores.mean())


# Run deterministic PAM. BUILD chooses initial representatives and SWAP replaces
# them only when the total distance decreases; asset indices resolve exact ties.
def fit_kmedoids(
    distance_matrix,
    number_of_clusters,
    maximum_iterations=100,
    tolerance=1e-12,
):
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    number_of_items = len(matrix)
    if not 2 <= number_of_clusters < number_of_items:
        raise ValueError("number_of_clusters must belong to [2, n-1]")

    first_medoid = int(np.argmin(matrix.sum(axis=0)))
    medoids = (first_medoid,)

    while len(medoids) < number_of_clusters:
        candidates = []
        for candidate in range(number_of_items):
            if candidate in medoids:
                continue
            proposal = tuple(sorted(medoids + (candidate,)))
            _, objective = assign_to_medoids(matrix, proposal)
            candidates.append((objective, candidate))

        # Python compares the objective first and the asset index second, which
        # gives a stable answer when two candidates have the same objective.
        _, selected = min(candidates)
        medoids = tuple(sorted(medoids + (selected,)))

    _, current_objective = assign_to_medoids(matrix, medoids)
    iterations = 0

    for iterations in range(1, maximum_iterations + 1):
        non_medoids = []
        for item in range(number_of_items):
            if item not in medoids:
                non_medoids.append(item)

        proposals = []
        for old_medoid in medoids:
            for new_medoid in non_medoids:
                proposal = []
                for medoid in medoids:
                    if medoid == old_medoid:
                        proposal.append(new_medoid)
                    else:
                        proposal.append(medoid)
                proposal = tuple(sorted(proposal))
                _, objective = assign_to_medoids(matrix, proposal)
                proposals.append((objective, proposal))

        best_objective, best_medoids = min(proposals)
        if best_objective >= current_objective - tolerance:
            break

        medoids = best_medoids
        current_objective = best_objective

    labels, objective = assign_to_medoids(matrix, medoids)
    return {
        "medoid_indices": medoids,
        "labels": labels,
        "objective": objective,
        "silhouette": silhouette_from_distance_matrix(matrix, labels),
        "iterations": iterations,
    }


# Convert array-based results to names and plain values so each partition is easy
# to inspect and can later be loaded without importing the clustering code.
def serialize_kmedoids_result(result, symbols):
    medoids = result["medoid_indices"]
    labels = result["labels"]
    clusters = {}
    assignments = {}

    for cluster in range(len(medoids)):
        clusters[str(cluster)] = []

    for position, symbol in enumerate(symbols):
        cluster = int(labels[position])
        assignments[symbol] = cluster
        clusters[str(cluster)].append(symbol)

    medoid_symbols = []
    for index in medoids:
        medoid_symbols.append(symbols[index])

    return {
        "number_of_clusters": len(medoids),
        "medoid_indices": list(medoids),
        "medoid_symbols": medoid_symbols,
        "assignments": assignments,
        "clusters": clusters,
        "objective": result["objective"],
        "silhouette": result["silhouette"],
        "pam_iterations": result["iterations"],
    }
