# tests/test_kernels.py
import numpy as np
import pytest

from buc import (
    build_dense_lut,
    fill_symmetric_filter_lut,
    precompute_mixing_recipes,
    compute_stage1_spurs_no_if2,
    compute_stage2_from_intermediates,
)


@pytest.mark.parametrize("grid_max,step", [(10.0, 1.0), (100.0, 2.0)])
def test_build_dense_lut_interpolates_linearly(grid_max, step):
    """
    Sanity check: build_dense_lut should essentially interpolate x_pts/y_pts
    onto the dense grid.

    We don't assume a particular index for mid-frequency (because the LUT
    grid is np.linspace(0, grid_max, n_bins)), but we *do* check that:
      - the first bin matches the first y-point
      - the last bin matches the last y-point
      - the LUT is monotonic between them
    """
    x_pts = np.array([0.0, grid_max], dtype=float)
    y_pts = np.array([0.0, -20.0], dtype=float)

    lut = build_dense_lut(x_pts, y_pts, grid_max, step, default_val=0.0)
    n_bins = int(grid_max / step) + 2
    assert lut.shape[0] == n_bins

    # Endpoints should match y_pts (within small tolerance)
    assert lut[0] == pytest.approx(0.0, abs=1e-3)
    assert lut[-1] == pytest.approx(-20.0, abs=0.5)

    # The LUT should be monotonically non-increasing (0 down to -20)
    diffs = np.diff(lut)
    # Allow tiny positive numerical noise
    assert np.all(diffs <= 1e-5 + 1e-8)


def test_fill_symmetric_filter_lut_passband_and_stopband():
    """
    For a symmetric power-law IF2 model:
    - passband around center should be ~insertion loss
    - far away should approach stop_floor
    - shape is symmetric.
    """
    center = 10e6
    bw = 4e6
    il_db = 1.0
    rolloff = 40.0
    stop_floor = 80.0
    step = 0.5e6

    n_bins = 81  # ~0..40 MHz
    lut = np.full(n_bins, stop_floor, dtype=np.float32)

    fill_symmetric_filter_lut(lut, center, bw, il_db, rolloff, stop_floor, step)

    idx_center = int(center / step)
    assert lut[idx_center] == pytest.approx(il_db, abs=1e-3)

    # Symmetry check a bit off-center
    delta = 1e6
    idx_lo = int((center - delta) / step)
    idx_hi = int((center + delta) / step)
    assert lut[idx_lo] == pytest.approx(lut[idx_hi], rel=1e-6)

    # Far out in stopband
    idx_far = int((center + 10 * bw) / step)
    idx_far = min(idx_far, n_bins - 1)
    assert lut[idx_far] >= il_db
    assert lut[idx_far] <= stop_floor + 1e-3  # clamped


def test_precompute_mixing_recipes_and_stage1_spurs_basic():
    """
    Check that the LO + mixer spur table combination produces
    a sensible recipe list and stage1 spur frequencies.
    """
    # Single LO carrier at 1 GHz, 0 dBc
    lo_comps = np.array([[1.0e9, 0.0]], dtype=np.float64)

    # Single spur term: m=1, n=1, base_rej=-30 dBc
    mxr_table = np.array([[1.0, 1.0, -30.0]], dtype=np.float64)

    drive_delta_db = 0.0
    scale_slope = 1.0
    scale_cap = 12.0
    max_order = 3

    recipes = precompute_mixing_recipes(
        lo_comps,
        mxr_table,
        search_mode=True,
        drive_delta_db=drive_delta_db,
        scale_slope=scale_slope,
        scale_cap=scale_cap,
        max_order=max_order,
        include_lo_feedthrough=False,
        lo_feedthrough_rej_db=0.0,
        dominant_only=False,
    )

    # Expect 4 sign combinations for (±LO ± IF)
    assert recipes.shape[0] == 4
    # gain should be LO (0 dBc) + base_rej(-30)
    assert np.allclose(recipes[:, 2], -30.0, atol=1e-6)

    # Stage-1 spur generation: f_if2 = |eff_lo + signed_n * f_if1|
    if1_freq = 100e6
    buf = np.zeros((16, 4), dtype=np.float64)
    count = compute_stage1_spurs_no_if2(recipes, if1_freq, buf)
    assert count == 4

    freqs = buf[:count, 0]
    # We know eff_lo = ±1GHz, signed_n=±1 => {900, 1100} MHz
    assert np.all(np.isin(freqs, [0.9e9, 1.1e9]))


def test_compute_stage2_simple_direct_spur():
    """
    Simple controlled case for compute_stage2_from_intermediates:
    one direct spur, no leakage spurs, flat RF & mask LUTs.

    We avoid the "main carrier skip" logic by placing the spur well away from
    rf_freq_desired (beyond the 100 Hz tolerance).
    """
    # No stage-1 leakage spurs
    stage1_spurs = np.zeros((0, 4), dtype=np.float64)

    # Single stage-2 recipe:
    # f_spur_rf = |eff_lo + signed_n * f_if2_desired|
    # Choose f_if2_desired = 5 Hz, eff_lo = 200 Hz => f_spur_rf = 205 Hz
    eff_lo = 200.0
    signed_n = 1.0
    gain = -30.0  # dBc at mixer-2 output
    m2_abs = 1.0
    n2_abs = 1.0
    is_desired = 0.0
    stage2_recipes = np.array([[eff_lo, signed_n, gain, m2_abs, n2_abs, is_desired]], dtype=np.float64)

    rf_freq_desired = 0.0   # "carrier" at 0 Hz
    f_if2_desired = 5.0     # intermediate frequency

    # Flat RF & mask LUTs over a large enough grid to include 205 Hz
    grid_step = 1.0
    n_bins = 512  # covers 0..511 Hz
    rf_lut = np.zeros(n_bins, dtype=np.float32)          # 0 dB attenuation
    mask_lut = np.full(n_bins, -60.0, dtype=np.float32)  # -60 dBc mask everywhere

    guard_db = 3.0
    noise_floor = -100.0
    rbw_hz = 1.0
    cross_stage_sum_max = 20
    max_spur_dbc = 0.0

    margin = compute_stage2_from_intermediates(
        stage1_spurs,
        stage2_recipes,
        rf_freq_desired,
        f_if2_desired,
        rf_lut,
        mask_lut,
        grid_step,
        guard_db,
        noise_floor,
        rbw_hz,
        cross_stage_sum_max,
        max_spur_dbc,
    )

    # Spur at 205 Hz → mask(-60) - gain(-30) - guard(3) = -33 dB
    assert margin == pytest.approx(-33.0, rel=1e-3)
