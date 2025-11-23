import numpy as np
from numba import jit

# --- LUT Builders ---

def build_dense_lut(x_pts, y_pts, grid_max_hz, grid_step_hz, default_val):
    n_bins = int(grid_max_hz / grid_step_hz) + 2
    lut = np.full(n_bins, default_val, dtype=np.float32)
    
    if len(x_pts) == 0:
        return lut

    # Pre-process for sloped masks (manual handling if not pure interp)
    # Since x_pts/y_pts here usually come from standard interp for RF filter,
    # this function is mostly for RF S21. 
    # Mask construction uses explicit _apply_mask_table logic in SpurEngine.
    
    grid_freqs = np.linspace(0, grid_max_hz, n_bins)
    interpolated = np.interp(grid_freqs, x_pts, y_pts)
    lut[:] = interpolated
    return lut

# --- JIT Kernels ---

@jit(nopython=True, fastmath=True)
def get_lut_val(f_hz, lut, step_hz):
    idx_float = f_hz / step_hz
    idx = int(idx_float)
    if idx < 0: return lut[0]
    if idx >= len(lut) - 1: return lut[-1]
    frac = idx_float - idx
    val = lut[idx] * (1.0 - frac) + lut[idx+1] * frac
    return val

@jit(nopython=True, fastmath=True)
def fill_symmetric_filter_lut(lut, center_hz, bw_hz, il_db, rolloff, stop_floor, step_hz):
    """
    Symmetric power-law IF2 model.
    """
    n_bins = len(lut)
    hbw = bw_hz / 2.0
    start_f = center_hz - bw_hz * 5.0
    stop_f = center_hz + bw_hz * 5.0
    idx_start = max(0, int(start_f / step_hz))
    idx_stop = min(n_bins, int(stop_f / step_hz))

    if il_db < 0.0:
        il_db = 0.0
    if stop_floor < il_db:
        stop_floor = il_db

    for i in range(idx_start, idx_stop):
        f = i * step_hz
        delta = abs(f - center_hz)
        if delta <= hbw:
            lut[i] = il_db
        else:
            ratio = delta / hbw
            if ratio < 1.0:
                ratio = 1.0
            atten = il_db + rolloff * np.log10(ratio)
            if atten > stop_floor:
                atten = stop_floor
            if atten < il_db:
                atten = il_db
            lut[i] = atten

@jit(nopython=True, fastmath=True)
def fill_scaled_s2p_lut(lut, center_hz, bw_hz, proto_x, proto_y, stop_floor, step_hz):
    """
    Scaled S2P prototype.
    """
    n_bins = len(lut)
    min_x, max_x = proto_x[0], proto_x[-1]
    min_freq = min_x * bw_hz + center_hz
    max_freq = max_x * bw_hz + center_hz
    idx_start = max(0, int(min_freq / step_hz))
    idx_stop = min(n_bins, int(max_freq / step_hz))

    if stop_floor < 0.0:
        stop_floor = -stop_floor

    for i in range(idx_start, idx_stop):
        f_grid = i * step_hz
        x_req = (f_grid - center_hz) / bw_hz
        val = np.interp(x_req, proto_x, proto_y)
        if val > stop_floor:
            val = stop_floor
        if val < 0.0:
            val = 0.0
        lut[i] = val

# --- CORE SPUR PHYSICS ---

@jit(nopython=True, fastmath=True)
def precompute_mixing_recipes(lo_comps, mxr_table, search_mode, 
                              drive_delta_db, scale_slope, scale_cap,
                              max_order,
                              include_lo_feedthrough=False,
                              lo_feedthrough_rej_db=0.0,
                              dominant_only=False):
    """
    Flattens LO Spectrum * Mixer Table into a single list of recipes.

    IMPORTANT FIXES:
    - No longer enumerate ±LO harmonics separately:
        eff_lo is always +m * f_LO.
    - For n > 0, enumerate ±n on the IF side (sum/diff products),
      EXCEPT for the special "pure IF" families (m = 0, n > 0), where
      |m f_LO ± n f_IF| collapses to the same |n| f_IF.  In that case
      emit only a single signed_n = +n recipe.
    - For n == 0, emit a single recipe (no sum/diff).

    This removes:
      * the old 4× duplication from (+LO, ±n) vs (−LO, ±n), and
      * the residual 2× duplication for (m = 0, ±n) that collapses
        to the same physical “pure IF” spur under abs().
    """
    n_lo = lo_comps.shape[0]
    n_mx = mxr_table.shape[0]
    max_k = 10 if search_mode else 99

    if search_mode:
        active_max_order = max_order if max_order < 10 else 10
    else:
        active_max_order = max_order

    # LO drive scaling
    correction = drive_delta_db * scale_slope
    if correction > scale_cap:
        correction = scale_cap
    elif correction < -scale_cap:
        correction = -scale_cap

    # Conservative size estimate (still safe even though we generate fewer)
    size_est = n_lo * n_mx * 4 + (1 if include_lo_feedthrough else 0)
    recipes = np.zeros((size_est, 6), dtype=np.float64)
    count = 0

    # Optional explicit LO feedthrough spur (separate from spur table entries)
    if include_lo_feedthrough:
        f_lo_main = lo_comps[0, 0]
        lvl_lo = lo_comps[0, 1] + lo_feedthrough_rej_db
        recipes[count, 0] = f_lo_main     # eff_lo
        recipes[count, 1] = 0.0           # signed_n
        recipes[count, 2] = lvl_lo        # gain
        recipes[count, 3] = 1.0           # m_abs
        recipes[count, 4] = 0.0           # n_abs
        recipes[count, 5] = 0.0           # is_desired (unused)
        count += 1

    # Phase-1 assumption: desired family is (1,1); the engine later
    # suppresses the desired path explicitly by geometry, so the
    # is_desired flag here is informational only.
    desired_m = 1
    desired_n = 1

    for k in range(n_lo):
        if search_mode and k > max_k:
            break

        f_lo_c = lo_comps[k, 0]
        p_lo_c = lo_comps[k, 1]

        is_main_lo = (k == 0)

        for i in range(n_mx):
            m = int(mxr_table[i, 0])
            n = int(mxr_table[i, 1])
            base_rej = mxr_table[i, 2]

            if m > active_max_order or n > active_max_order:
                continue

            # Dominant spur pruning
            if dominant_only:
                is_dom = False
                if ((m == 1 and n == 1) or
                    (m == 1 and n == 2) or
                    (m == 2 and n == 1) or
                    (m == 2 and n == 2) or
                    (m == 3 and n == 1)):
                    is_dom = True
                if not is_dom:
                    continue

            scaled_rej = base_rej - correction
            lvl_fixed = p_lo_c + scaled_rej

            # n == 0 : pure LO term → one recipe only (no ±n)
            if n == 0:
                if count >= size_est:
                    break

                eff_lo = m * f_lo_c  # always +m * f_LO

                recipes[count, 0] = eff_lo
                recipes[count, 1] = 0.0               # signed_n
                recipes[count, 2] = lvl_fixed
                recipes[count, 3] = float(m)
                recipes[count, 4] = float(n)
                # is_desired flag is informational only and currently unused
                recipes[count, 5] = 1.0 if (is_main_lo and
                                            m == desired_m and
                                            n == desired_n) else 0.0
                count += 1
                continue

            # n > 0 : sum/diff products; LO sign is not enumerated
            eff_lo = m * f_lo_c  # +m * f_LO

            if m == 0:
                # Special-case pure IF families (m = 0, n > 0):
                # |0 * f_LO ± n * f_IF| → |n| f_IF, so ±n are the same
                # physical spur under abs(). Emit only +n.
                if count >= size_est:
                    break

                signed_n = float(n)
                # This can never be the desired (1,1) family, so is_des = 0.0
                is_des = 0.0

                recipes[count, 0] = eff_lo
                recipes[count, 1] = signed_n
                recipes[count, 2] = lvl_fixed
                recipes[count, 3] = float(m)
                recipes[count, 4] = float(n)
                recipes[count, 5] = is_des
                count += 1
            else:
                # General case: keep ±n (sum/diff) for m != 0
                for s_in in (-1, 1):
                    if count >= size_est:
                        break

                    signed_n = s_in * n

                    # Informational desired-flag only; engine does not rely on this
                    is_des = 1.0 if (is_main_lo and
                                     m == desired_m and
                                     n == desired_n) else 0.0

                    recipes[count, 0] = eff_lo
                    recipes[count, 1] = signed_n
                    recipes[count, 2] = lvl_fixed
                    recipes[count, 3] = float(m)
                    recipes[count, 4] = float(n)
                    recipes[count, 5] = is_des
                    count += 1

    return recipes[:count]

@jit(nopython=True, fastmath=True)
def compute_stage1_spurs_no_if2(
    stage1_recipes,   # [eff_lo, signed_n, gain, m_abs, n_abs, is_desired]
    if1_freq,
    result_buffer     # shape (N, 4): [f_if2, lvl_pre_if2, m1_abs, n1_abs]
):
    """
    IF2-agnostic Stage-1 spur generation.
    Computes f_if2 = | eff_lo + signed_n * if1_freq |
    
    Purely algebraic; does NOT attempt to identify or suppress desired paths.
    Desired path suppression is handled by the caller (SpurEngine) based on
    exact geometric constraints.
    """
    count = 0
    max_buffer = result_buffer.shape[0]
    n_recipes = stage1_recipes.shape[0]

    for i in range(n_recipes):
        eff_lo = stage1_recipes[i, 0]
        signed_n = stage1_recipes[i, 1]
        gain = stage1_recipes[i, 2]
        m_abs = stage1_recipes[i, 3]
        n_abs = stage1_recipes[i, 4]
        
        # Algebraic mixing product
        f_if2 = abs(eff_lo + signed_n * if1_freq)

        if count >= max_buffer:
            return -count  # overflow

        result_buffer[count, 0] = f_if2
        result_buffer[count, 1] = gain
        result_buffer[count, 2] = m_abs
        result_buffer[count, 3] = n_abs
        count += 1

    return count

@jit(nopython=True, fastmath=True)
def compute_stage2_from_intermediates(
    stage1_spurs,       # [f_if2, lvl_after_if2, m1_abs, n1_abs]
    stage2_recipes,     # [eff_lo, signed_n, gain, m2_abs, n2_abs, is_desired]
    rf_freq_desired,
    f_if2_desired,
    rf_lut,
    mask_lut,
    grid_step,
    guard_db,
    noise_floor_dbc,
    rbw_hz,
    cross_stage_sum_max,
    max_spur_dbc        # Clamp for spur levels
):
    """
    Stage-2 evaluation with explicit clamp and cross-stage check.
    Leakage math updated to be physically additive (dB+dB).
    """
    min_margin = 999.0
    n_recipes = stage2_recipes.shape[0]
    n_s1 = stage1_spurs.shape[0]
    n_mask = mask_lut.shape[0]
    
    pow_bins = np.zeros(n_mask)
    bins_per_rbw = int(rbw_hz / grid_step)
    if bins_per_rbw < 1:
        bins_per_rbw = 1
    
    # --- 1. Direct IF2 -> RF spurs ---
    # Phase-1: Desired path is fixed to (1,1)
    m1_abs_direct = 1.0
    n1_abs_direct = 1.0

    for i in range(n_recipes):
        eff_lo = stage2_recipes[i, 0]
        signed_n = stage2_recipes[i, 1]
        gain = stage2_recipes[i, 2]
        m2_abs = stage2_recipes[i, 3]
        n2_abs = stage2_recipes[i, 4]
        
        total_order = m1_abs_direct + n1_abs_direct + m2_abs + n2_abs
        if total_order > cross_stage_sum_max:
            continue
            
        f_spur_rf = abs(eff_lo + signed_n * f_if2_desired)
        
        # Skip main carrier
        tol = rf_freq_desired * 1e-6
        if tol < 100.0: tol = 100.0
        if abs(f_spur_rf - rf_freq_desired) < tol: 
            continue

        idx = int(f_spur_rf / grid_step)
        if idx < 0 or idx >= n_mask: continue

        atten_rf = get_lut_val(f_spur_rf, rf_lut, grid_step)
        final_lvl = gain - atten_rf
        
        if final_lvl < noise_floor_dbc: continue
        
        limit = mask_lut[idx]
        margin_line = limit - final_lvl - guard_db
        if margin_line < min_margin: min_margin = margin_line

        p_lin = 10.0 ** (final_lvl / 10.0)
        pow_bins[idx] += p_lin

    # --- 2. Stage-1 leakage spurs -> Stage-2 ---
    for i_s1 in range(n_s1):
        f_input = stage1_spurs[i_s1, 0]
        l_input = stage1_spurs[i_s1, 1] # dBc at IF2
        m1_abs = stage1_spurs[i_s1, 2]
        n1_abs = stage1_spurs[i_s1, 3]
        
        # Configurable clamp: only active if max_spur_dbc is negative (e.g. -10)
        # If max_spur_dbc is 0.0, no clamp is applied.
        if max_spur_dbc < 0.0 and l_input > max_spur_dbc:
            l_input = max_spur_dbc

        for i in range(n_recipes):
            eff_lo = stage2_recipes[i, 0]
            signed_n = stage2_recipes[i, 1]
            gain = stage2_recipes[i, 2]
            m2_abs = stage2_recipes[i, 3]
            n2_abs = stage2_recipes[i, 4]
            
            total_order = m1_abs + n1_abs + m2_abs + n2_abs
            if total_order > cross_stage_sum_max:
                continue
            
            f_final_rf = abs(eff_lo + signed_n * f_input)
            
            idx = int(f_final_rf / grid_step)
            if idx < 0 or idx >= n_mask: continue

            # Leakage model: Stage-1 level (dB) + Stage-2 gain (dB)
            lvl_rf = l_input + gain
            
            atten_rf = get_lut_val(f_final_rf, rf_lut, grid_step)
            final_lvl = lvl_rf - atten_rf
            
            if final_lvl < noise_floor_dbc: continue
            
            limit = mask_lut[idx]
            margin_line = limit - final_lvl - guard_db
            if margin_line < min_margin: min_margin = margin_line

            p_lin = 10.0 ** (final_lvl / 10.0)
            pow_bins[idx] += p_lin

    # --- 3. RBW aggregation ---
    if bins_per_rbw == 1:
        for idx in range(n_mask):
            if pow_bins[idx] <= 0.0:
                continue
            lvl_db = 10.0 * np.log10(pow_bins[idx])
            limit = mask_lut[idx]
            margin_bin = limit - lvl_db - guard_db
            if margin_bin < min_margin:
                min_margin = margin_bin
    else:
        window_power = 0.0
        for idx in range(n_mask):
            window_power += pow_bins[idx]
            if idx >= bins_per_rbw:
                window_power -= pow_bins[idx - bins_per_rbw]
            if window_power <= 0.0:
                continue
            lvl_db = 10.0 * np.log10(window_power)
            limit = mask_lut[idx]
            margin_bin = limit - lvl_db - guard_db
            if margin_bin < min_margin:
                min_margin = margin_bin

    return min_margin