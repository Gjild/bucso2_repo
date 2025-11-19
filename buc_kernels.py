import numpy as np
from numba import jit

# --- LUT Builders ---

def build_dense_lut(x_pts, y_pts, grid_max_hz, grid_step_hz, default_val):
    """ Converts X/Y points into a dense array indexable by int(f / step). """
    n_bins = int(grid_max_hz / grid_step_hz) + 1
    lut = np.full(n_bins, default_val, dtype=np.float32)
    
    if len(x_pts) == 0:
        return lut

    grid_freqs = np.linspace(0, grid_max_hz, n_bins)
    interpolated = np.interp(grid_freqs, x_pts, y_pts)
    lut[:] = interpolated
    return lut

# --- JIT Kernels ---

@jit(nopython=True, fastmath=True)
def get_lut_val(f_hz, lut, step_hz):
    """ O(1) Lookup """
    idx = int(f_hz / step_hz)
    if idx < 0: return lut[0]
    if idx >= len(lut): return lut[-1]
    return lut[idx]

@jit(nopython=True, fastmath=True)
def fill_symmetric_filter_lut(lut, center_hz, bw_hz, il_db, rolloff, stop_floor, step_hz):
    """ Analytic fill for Symmetric Powerlaw IF2 filter. """
    n_bins = len(lut)
    hbw = bw_hz / 2.0
    
    # Optimization bounds
    start_f = center_hz - bw_hz * 5
    stop_f = center_hz + bw_hz * 5
    idx_start = max(0, int(start_f / step_hz))
    idx_stop = min(n_bins, int(stop_f / step_hz))
    
    for i in range(idx_start, idx_stop):
        f = i * step_hz
        delta = abs(f - center_hz)
        
        if delta <= hbw:
            lut[i] = il_db
        else:
            ratio = delta / hbw
            atten = il_db + rolloff * np.log10(ratio)
            if atten > -stop_floor: 
                lut[i] = -stop_floor
            else:
                lut[i] = atten

@jit(nopython=True, fastmath=True)
def fill_scaled_s2p_lut(lut, center_hz, bw_hz, proto_x, proto_y, stop_floor, step_hz):
    """ 
    Fills LUT by interpolating a normalized prototype.
    """
    n_bins = len(lut)
    
    # Optimization bounds
    min_x = proto_x[0]
    max_x = proto_x[-1]
    
    min_freq = min_x * bw_hz + center_hz
    max_freq = max_x * bw_hz + center_hz
    
    idx_start = max(0, int(min_freq / step_hz))
    idx_stop = min(n_bins, int(max_freq / step_hz))
    
    for i in range(idx_start, idx_stop):
        f_grid = i * step_hz
        x_req = (f_grid - center_hz) / bw_hz
        
        # Linear Interp on Prototype
        val = np.interp(x_req, proto_x, proto_y)
        
        if val > -stop_floor:
             lut[i] = -stop_floor
        else:
             lut[i] = val

# --- CORE SPUR PHYSICS (SPLIT KERNELS) ---

@jit(nopython=True, fastmath=True)
def compute_stage1_intermediates(
    lo1_comps,          # [[freq, dBc], ...]
    lo1_freq,           # Fundamental LO
    if1_freq,
    mxr1_table,         # [[m, n, rej], ...]
    if2_side_high,      # bool: True if Sum, False if Diff
    if2_lut,
    grid_step,
    noise_floor_dbc,
    search_mode         # bool: if True, restrict orders
):
    """
    Calculates all significant spur tones entering Stage 2.
    Returns a simplified array: [[freq_hz, level_dBm_equiv], ...]
    """
    # Increased buffer size to prevent silent truncation
    # Added sentinel mechanism for overflow
    MAX_SPURS = 512
    results = np.zeros((MAX_SPURS, 2), dtype=np.float64)
    count = 0
    
    if if2_side_high:
        f_if2_desired = lo1_freq + if1_freq
    else:
        f_if2_desired = abs(lo1_freq - if1_freq)

    n_lo1 = lo1_comps.shape[0]
    n_mx1 = mxr1_table.shape[0]
    
    max_k = 3 if search_mode else 99
    max_ord = 3 if search_mode else 99

    for k1 in range(n_lo1):
        if search_mode and k1 > max_k: break
        
        f_lo1_c = lo1_comps[k1, 0]
        p_lo1_c = lo1_comps[k1, 1]
        
        for i1 in range(n_mx1):
            m1 = int(mxr1_table[i1, 0])
            n1 = int(mxr1_table[i1, 1])
            
            if search_mode and (m1 > max_ord or n1 > max_ord): continue
            
            rej1 = mxr1_table[i1, 2]
            
            for s_lo1 in (-1, 1):
                for s_if1 in (-1, 1):
                    f_spur_if2 = abs(s_lo1 * m1 * f_lo1_c + s_if1 * n1 * if1_freq)
                    
                    # Strict Integer Check for Desired Path
                    is_desired = (k1==0 and m1==1 and n1==1)
                    if is_desired:
                        if abs(f_spur_if2 - f_if2_desired) < 1.0:
                            continue

                    lvl_s1 = p_lo1_c + rej1 
                    
                    atten_if2 = get_lut_val(f_spur_if2, if2_lut, grid_step)
                    lvl_input_to_m2 = lvl_s1 - atten_if2
                    
                    if lvl_input_to_m2 < noise_floor_dbc: continue
                    
                    # Check for Buffer Overflow
                    if count >= MAX_SPURS:
                        # Set sentinel to signal failure
                        results[0, 0] = -1.0 
                        return results
                        
                    results[count, 0] = f_spur_if2
                    results[count, 1] = lvl_input_to_m2
                    count += 1
                        
    return results[:count]

@jit(nopython=True, fastmath=True)
def compute_stage2_from_intermediates(
    stage1_spurs,       # [[freq, lvl], ...] from previous kernel
    lo2_comps,          # [[freq, dBc], ...]
    lo2_freq,           # Fund LO2
    rf_freq,            # Desired RF
    mxr2_table,         # [[m, n, rej], ...]
    rf_side_high,       # bool: True if Sum, False if Diff
    f_if2_desired,      # The main IF2 signal frequency
    rf_lut,
    mask_lut,
    grid_step,
    guard_db,
    noise_floor_dbc,
    search_mode         # bool
):
    """
    Mixes (Stage1_Spurs + Desired_IF2) with LO2 -> RF.
    Checks against masks.
    """
    # 2.1: Check for error sentinel from Stage 1
    if stage1_spurs.shape[0] > 0 and stage1_spurs[0, 0] == -1.0:
        return -999.0 # Propagate error

    min_margin = 999.0
    
    n_lo2 = lo2_comps.shape[0]
    n_mx2 = mxr2_table.shape[0]
    n_s1 = stage1_spurs.shape[0]
    
    max_k = 3 if search_mode else 99
    max_ord = 3 if search_mode else 99

    # 1. Process Desired IF2 Signal -> Stage 2 Spurs
    for k2 in range(n_lo2):
        if search_mode and k2 > max_k: break
        
        f_lo2_c = lo2_comps[k2, 0]
        p_lo2_c = lo2_comps[k2, 1]
        
        for i2 in range(n_mx2):
            m2 = int(mxr2_table[i2, 0])
            n2 = int(mxr2_table[i2, 1])
            
            if search_mode and (m2 > max_ord or n2 > max_ord): continue
            
            rej2 = mxr2_table[i2, 2]
            
            for s_lo2 in (-1, 1):
                for s_if2 in (-1, 1):
                    f_spur_rf = abs(s_lo2 * m2 * f_lo2_c + s_if2 * n2 * f_if2_desired)
                    
                    if k2==0 and m2==1 and n2==1 and abs(f_spur_rf - rf_freq) < 1.0:
                        continue
                        
                    lvl = p_lo2_c + rej2 
                    atten_rf = get_lut_val(f_spur_rf, rf_lut, grid_step)
                    final_lvl = lvl - atten_rf
                    
                    if final_lvl < noise_floor_dbc: continue
                    
                    limit = get_lut_val(f_spur_rf, mask_lut, grid_step)
                    margin = limit - final_lvl - guard_db
                    
                    if margin < min_margin:
                        min_margin = margin

    # 2. Process Stage 1 Spurs (Leakage) -> Stage 2
    for i_s1 in range(n_s1):
        f_input = stage1_spurs[i_s1, 0]
        l_input = stage1_spurs[i_s1, 1]
        
        for k2 in range(n_lo2):
            if search_mode and k2 > max_k: break
            
            f_lo2_c = lo2_comps[k2, 0]
            p_lo2_c = lo2_comps[k2, 1]
            
            # Optimization: Only mix spurious inputs with n2=1
            for i2 in range(n_mx2):
                n2 = int(mxr2_table[i2, 1])
                if n2 != 1: continue 
                
                m2 = int(mxr2_table[i2, 0])
                rej2 = mxr2_table[i2, 2]
                
                for s_lo2 in (-1, 1):
                    for s_if2 in (-1, 1):
                        f_final_rf = abs(s_lo2 * m2 * f_lo2_c + s_if2 * n2 * f_input)
                        
                        lvl_rf = l_input + p_lo2_c + rej2
                        atten_rf = get_lut_val(f_final_rf, rf_lut, grid_step)
                        final_lvl = lvl_rf - atten_rf
                        
                        if final_lvl < noise_floor_dbc: continue
                        
                        limit = get_lut_val(f_final_rf, mask_lut, grid_step)
                        margin = limit - final_lvl - guard_db
                        
                        if margin < min_margin:
                            min_margin = margin

    return min_margin