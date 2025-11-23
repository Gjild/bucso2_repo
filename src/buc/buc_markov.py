import numpy as np
import pandas as pd
import json

def markov_lock_summary(df_policy: pd.DataFrame, 
                        markov_matrix: np.ndarray, 
                        out_path: str = "markov_lock_summary.json"):
    """
    Diagnostic-only Markov-weighted lock-time summary.
    """
    if markov_matrix.size == 0:
        return

    df_policy = df_policy.sort_values("tile_id")
    tile_ids = df_policy["tile_id"].to_numpy()
    lock_times = df_policy["lock_time_ms_tile"].to_numpy()

    n = len(tile_ids)
    if markov_matrix.shape != (n, n):
        print("Markov matrix shape mismatch; skipping Markov summary.")
        return

    # Stationary distribution via power iteration
    dist = np.ones(n) / n
    for _ in range(1000):
        dist_new = dist @ markov_matrix
        if np.max(np.abs(dist_new - dist)) < 1e-9:
            dist = dist_new
            break
        dist = dist_new

    expected_lock_time = float(np.dot(dist, lock_times))

    summary = {
        "num_tiles": int(n),
        "tile_ids": tile_ids.tolist(),
        "lock_time_ms_tile": lock_times.tolist(),
        "stationary_distribution": dist.tolist(),
        "expected_lock_time_ms_stationary": expected_lock_time,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Markov lock-time summary written to {out_path}")