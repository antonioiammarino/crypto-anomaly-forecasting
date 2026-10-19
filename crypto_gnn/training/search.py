import random


MAJORITY_CANDIDATE = {
    "identifier": "majority_training_class",
    "experimental_arm": "majority",
    "architecture": "majority",
    "edge_mode": "temporal",
    "visibility_window": None,
    "temporal_window_hours": 1,
    "hidden_channels": 0,
    "attention_heads": 0,
    "dropout": 0.0,
    "learning_rate": 0.0,
    "weight_decay": 0.0,
}


# Generate the deterministic random search for one neural arm. Each graph window or
# MLP temporal window receives its default model before the remaining candidates are
# sampled from the complete hyperparameter space.
def build_candidates(search_space, experimental_arm):
    if experimental_arm == "gat_temporal":
        architecture = "gat"
        structures = [("temporal", None, 1)]
    elif experimental_arm == "gat_visibility":
        architecture = "gat"
        structures = []
        for window in search_space["visibility_windows"]:
            structures.append(("temporal_visibility", window, 1))
    else:
        architecture = "mlp"
        structures = []
        for window in search_space["mlp_window_hours"]:
            structures.append(("temporal", None, window))

    if architecture == "gat":
        default_heads = 4
        head_options = search_space["attention_heads"]
    else:
        default_heads = 1
        head_options = (1,)

    payloads = []
    for edge_mode, visibility_window, temporal_window in structures:
        payloads.append(
            {
                "experimental_arm": experimental_arm,
                "architecture": architecture,
                "edge_mode": edge_mode,
                "visibility_window": visibility_window,
                "temporal_window_hours": temporal_window,
                "hidden_channels": 16,
                "attention_heads": default_heads,
                "dropout": 0.3,
                "learning_rate": 0.005,
                "weight_decay": 0.0005,
            }
        )

    # Explicit loops keep the order clear. The order must remain stable because the
    # seeded random sample operates on this complete list.
    complete_space = []
    for structure in structures:
        for hidden_channels in search_space["hidden_channels"]:
            for attention_heads in head_options:
                for dropout in search_space["dropout"]:
                    for learning_rate in search_space["learning_rate"]:
                        for weight_decay in search_space["weight_decay"]:
                            edge_mode, visibility_window, temporal_window = structure
                            complete_space.append(
                                {
                                    "experimental_arm": experimental_arm,
                                    "architecture": architecture,
                                    "edge_mode": edge_mode,
                                    "visibility_window": visibility_window,
                                    "temporal_window_hours": temporal_window,
                                    "hidden_channels": hidden_channels,
                                    "attention_heads": attention_heads,
                                    "dropout": dropout,
                                    "learning_rate": learning_rate,
                                    "weight_decay": weight_decay,
                                }
                            )

    remaining = []
    for payload in complete_space:
        if payload not in payloads:
            remaining.append(payload)

    additional_count = search_space["candidates_per_neural_arm"] - len(payloads)
    generator = random.Random(f"{search_space['search_seed']}:{experimental_arm}")
    payloads.extend(generator.sample(remaining, additional_count))

    candidates = []
    for i, payload in enumerate(payloads, start=1):
        candidate = {"identifier": f"{experimental_arm}_{i}"}
        candidate.update(payload)
        candidates.append(candidate)
    return candidates
