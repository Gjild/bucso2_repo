# tests/test_spur_engine.py
import numpy as np
import pytest

from buc import FilterModel, Tile
from buc import SpurEngine


def test_early_reject_image_in_if2_passband(spur_engine: SpurEngine, cfg):
    """
    _early_reject_chain should reject a Sum-Sum mapping whose IF2 image
    falls inside the IF2 passband, but accept a geometry with the image far outside.

    This is a geometric unit test of the early-reject logic.
    """
    engine = spur_engine
    tile = cfg.tiles[0]

    # Narrow IF2 window around 3 GHz
    if2 = FilterModel(center_hz=3.0e9, bw_hz=100.0e6, model_type=0)

    # Sum-Sum: desired = LO1 + IF1, image = |LO1 - IF1|
    # Pick LO1 so that the image = IF2 center (inside passband)
    lo1_in = tile.if1_center_hz + if2.center_hz
    lo2_far = 10.0e9  # well away from RF passband

    assert engine._early_reject_chain(tile, if2, lo1_in, lo2_far, high_side=False)

    # Now push the image far outside passband: image = IF2 + 10*BW
    lo1_out = tile.if1_center_hz + if2.center_hz + 10 * if2.bw_hz
    assert not engine._early_reject_chain(tile, if2, lo1_out, lo2_far, high_side=False)


def test_build_policy_for_if2_returns_entry_per_tile(spur_engine: SpurEngine, cfg):
    """
    For a reasonable IF2 placement, build_policy_for_if2 should either:
      - produce a consistent entry per tile, OR
      - clearly signal "no solution" via sentinel margin and empty entries.

    When entries exist we additionally check:
      - number of entries == number of tiles
      - LO sides are consistent (Sum–Sum or Diff–Diff only).
    """
    # To keep runtime practical, trim the tile set for this test
    cfg.tiles = cfg.tiles[:8]
    engine = spur_engine

    if2_cfg = cfg.yaml_data["if2_model"]
    center_min_raw, center_max_raw = if2_cfg["center_range_hz"]
    center_min = float(center_min_raw)
    center_max = float(center_max_raw)

    bw_min = float(if2_cfg["min_bw_hz"])
    bw_max = float(if2_cfg["max_bw_hz"])

    # Mid-range IF2 candidate
    if2 = FilterModel(
        center_hz=0.5 * (center_min + center_max),
        bw_hz=0.5 * (bw_min + bw_max),
        model_type=0,
        passband_il=float(if2_cfg["passband_il_db"]),
        rolloff=float(if2_cfg["rolloff_db_per_dec"]),
        stop_floor=float(if2_cfg["stop_floor_db"]),
    )

    worst_margin, total_lock, retunes, entries = engine.build_policy_for_if2(
        if2,
        search_mode=True,
        compute_brittleness=False,
        stop_if_margin_below=None,
    )

    # Case 1: engine found no viable LO/IF2 mapping for this candidate.
    # It should then use the sentinel "no solution" pattern: empty entries
    # and very negative margin.
    if not entries:
        assert worst_margin <= -900.0
        # retunes should be a sane integer (typically 0 in this case)
        assert isinstance(retunes, int)
        return

    # Case 2: Valid policy exists. Now do the structural checks.
    assert len(entries) == len(cfg.tiles)

    tile_ids = sorted(e.tile_id for e in entries)
    assert tile_ids == [t.id for t in cfg.tiles]

    # Sides must be "low" or "high" and consistent across LO1 / LO2
    for e in entries:
        assert e.side in ("low", "high")
        assert e.lo1.side == e.side
        assert e.lo2.side == e.side

    # If we got here, we should not have the sentinel "no solution" margin
    assert worst_margin > -900.0


def test_evaluate_policy_is_deterministic(spur_engine: SpurEngine, cfg):
    """
    For a fixed IF2 filter and config, evaluate_policy should be deterministic:
      - identical scores and expected lock times across calls.
    """
    # Small subset to keep runtime down
    cfg.tiles = cfg.tiles[:6]
    engine = spur_engine

    if2_cfg = cfg.yaml_data["if2_model"]
    center_min_raw, center_max_raw = if2_cfg["center_range_hz"]
    center_min = float(center_min_raw)
    center_max = float(center_max_raw)

    bw_min = float(if2_cfg["min_bw_hz"])
    bw_max = float(if2_cfg["max_bw_hz"])

    if2 = FilterModel(
        center_hz=0.5 * (center_min + center_max),
        bw_hz=0.5 * (bw_min + bw_max),
        model_type=0,
        passband_il=float(if2_cfg["passband_il_db"]),
        rolloff=float(if2_cfg["rolloff_db_per_dec"]),
        stop_floor=float(if2_cfg["stop_floor_db"]),
    )

    score1, lock1 = engine.evaluate_policy(if2, search_mode=True)
    score2, lock2 = engine.evaluate_policy(if2, search_mode=True)

    assert np.isfinite(score1)
    assert np.isfinite(score2)
    assert np.isclose(score1, score2, atol=1e-6)
    assert np.isclose(lock1, lock2, atol=1e-6)
