# tests/test_visuals_and_markov.py
from pathlib import Path

import json
import numpy as np
import pandas as pd

from buc import plot_margin_heatmap
from buc import markov_lock_summary


def test_plot_margin_heatmap_creates_file(tmp_path: Path):
    """
    Basic smoke test: plotting a small synthetic policy dataframe
    should produce a PNG file without raising.
    """
    df = pd.DataFrame(
        {
            "tile_id": [0, 1, 2, 3],
            "if1_center_hz": [1.0e9, 1.1e9, 1.0e9, 1.1e9],
            "rf_center_hz": [28.0e9, 28.0e9, 29.0e9, 29.0e9],
            "spur_margin_db": [5.0, 10.0, 15.0, 20.0],
        }
    )

    out_file = tmp_path / "heatmap_margin.png"
    plot_margin_heatmap(df, filename=str(out_file))

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_markov_lock_summary_generates_json(tmp_path: Path):
    """
    markov_lock_summary should write a JSON file with expected keys when
    given a consistent Markov matrix and policy dataframe.
    """
    df_policy = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "lock_time_ms_tile": [10.0, 30.0],
        }
    )

    # Simple 2-state Markov chain
    markov_matrix = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)

    out_path = tmp_path / "markov_summary.json"
    markov_lock_summary(df_policy, markov_matrix, out_path=str(out_path))

    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert "expected_lock_time_ms_stationary" in data
    assert data["num_tiles"] == 2
    assert len(data["tile_ids"]) == 2
