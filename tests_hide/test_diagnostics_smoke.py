# tests/test_diagnostics_smoke.py
import pandas as pd

from buc import generate_spur_ledger
from buc import FilterModel


def test_generate_spur_ledger_smoke(spur_engine, cfg):
    """
    Smoke test: generate_spur_ledger should return a DataFrame (possibly empty)
    without raising, when given a simple, manually constructed policy row.
    """
    engine = spur_engine
    tile = cfg.tiles[0]

    if2_cfg = cfg.yaml_data["if2_model"]
    center_min_raw, center_max_raw = if2_cfg["center_range_hz"]
    center_min = float(center_min_raw)
    center_max = float(center_max_raw)

    if2 = FilterModel(
        center_hz=0.5 * (center_min + center_max),
        bw_hz=float(if2_cfg["min_bw_hz"]),
        model_type=0,
        passband_il=float(if2_cfg["passband_il_db"]),
        rolloff=float(if2_cfg["rolloff_db_per_dec"]),
        stop_floor=float(if2_cfg["stop_floor_db"]),
    )

    # Minimal fake policy row consistent with what generate_spur_ledger expects
    policy_row = {
        "lo1_hz": if2.center_hz - tile.if1_center_hz,  # Sum-Sum mapping
        "lo2_hz": tile.rf_center_hz - if2.center_hz,
        "lo1_side": "low",
    }
    policy_row = pd.Series(policy_row)

    df_ledger = generate_spur_ledger(
        engine,
        tile_id=tile.id,
        policy_row=policy_row,
        if2_filter=if2,
        report_threshold_db=10.0,
    )

    # No strict assertion on content; just ensure it is a DataFrame and has the expected columns if non-empty.
    assert isinstance(df_ledger, pd.DataFrame)
    if not df_ledger.empty:
        for col in [
            "Source_Stage",
            "Stage",
            "Freq_GHz",
            "Margin_dB",
            "Level_dBm_equiv",
            "Limit_dBc",
            "Formula",
        ]:
            assert col in df_ledger.columns
