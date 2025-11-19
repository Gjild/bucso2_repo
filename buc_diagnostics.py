import pandas as pd
import numpy as np
from buc_structures import Tile, FilterModel, LO_Candidate
from buc_kernels import get_lut_val

def generate_spur_ledger(engine, tile_id, policy_row, if2_filter: FilterModel):
    """
    Re-evaluates a specific Tile+LO configuration to generate a detailed list of spurs.
    """
    tile = next(t for t in engine.cfg.tiles if t.id == tile_id)
    
    lo1_freq = policy_row['lo1_hz']
    lo2_freq = policy_row['lo2_hz']
    
    lo1_spec = engine._get_lo_spectrum(engine.hw.lo1_def, lo1_freq)
    lo2_spec = engine._get_lo_spectrum(engine.hw.lo2_def, lo2_freq)
    
    # Explicit Side Detection from Policy
    # Side string is "low" or "high".
    # Low Side: F_out = LO + IF (Sum) or |LO-IF| if LO < IF.
    # Architecture spec says:
    # Sum = Non-Inverting (Side usually depends on freq, but logic is fixed to Sum/Diff)
    # Diff = Non-Inverting (if High Side)
    
    # In engine we mapped:
    # high_side=False -> Sum (Non-Invert)
    # high_side=True  -> Diff (High Side, Non-Invert)
    
    st1_mode = "SUM" if policy_row['lo1_side'] == 'low' else "DIFF"
    st2_mode = "SUM" if policy_row['lo2_side'] == 'low' else "DIFF"

    ledger = []

    # --- Helper to create rows ---
    def add_entry(stage, formula, freq_hz, level_db, limit_db):
        margin = limit_db - level_db - engine.cfg.yaml_data['constraints']['guard_margin_db']
        ledger.append({
            "Stage": stage,
            "Formula": formula,
            "Freq_GHz": freq_hz / 1e9,
            "Level_dBm_equiv": level_db, 
            "Limit_dBc": limit_db,
            "Margin_dB": margin
        })

    # --- STAGE 1 ANALYSIS ---
    f_if2_desired = if2_filter.center_hz
    
    for k in range(len(lo1_spec)):
        f_lo_c = lo1_spec[k, 0]
        p_lo_c = lo1_spec[k, 1]
        
        for row in engine.hw.mixer1.spur_table_np:
            m, n, rej = int(row[0]), int(row[1]), row[2]
            
            for s_lo in [-1, 1]:
                for s_if in [-1, 1]:
                    f_spur_if2 = abs(s_lo * m * f_lo_c + s_if * n * tile.if1_center_hz)
                    
                    if k==0 and m==1 and n==1 and abs(f_spur_if2 - f_if2_desired) < 1000:
                        continue

                    lvl = p_lo_c + rej
                    atten_if2 = get_lut_val(f_spur_if2, engine.if2_lut_buffer, engine.grid_step)
                    lvl -= atten_if2
                    
                    if lvl < -100: continue
                    
                    # Map to RF via Desired Stage 2 Path
                    if st2_mode == "SUM":
                        f_rf_equiv = lo2_freq + f_spur_if2
                    else:
                        f_rf_equiv = abs(lo2_freq - f_spur_if2)
                    
                    atten_rf = get_lut_val(f_rf_equiv, engine.rf_lut, engine.grid_step)
                    lvl -= atten_rf
                    
                    limit = get_lut_val(f_rf_equiv, engine.mask_lut, engine.grid_step)
                    
                    if (limit - lvl) < 10.0: 
                        formula = f"({m}*LO1{'H' if k>0 else ''} {s_lo:+} {n}*IF1 {s_if:+})"
                        add_entry(1, formula, f_rf_equiv, lvl, limit)

    # --- STAGE 2 ANALYSIS ---
    for k in range(len(lo2_spec)):
        f_lo_c = lo2_spec[k, 0]
        p_lo_c = lo2_spec[k, 1]
        
        for row in engine.hw.mixer2.spur_table_np:
            m, n, rej = int(row[0]), int(row[1]), row[2]
            
            for s_lo in [-1, 1]:
                for s_if in [-1, 1]:
                    f_spur_rf = abs(s_lo * m * f_lo_c + s_if * n * f_if2_desired)
                    
                    if k==0 and m==1 and n==1 and abs(f_spur_rf - tile.rf_center_hz) < 1000:
                        continue
                        
                    lvl = p_lo_c + rej
                    atten_rf = get_lut_val(f_spur_rf, engine.rf_lut, engine.grid_step)
                    lvl -= atten_rf
                    
                    if lvl < -100: continue
                    
                    limit = get_lut_val(f_spur_rf, engine.mask_lut, engine.grid_step)
                    
                    if (limit - lvl) < 10.0:
                        formula = f"({m}*LO2{'H' if k>0 else ''} {s_lo:+} {n}*IF2 {s_if:+})"
                        add_entry(2, formula, f_spur_rf, lvl, limit)

    df = pd.DataFrame(ledger)
    if not df.empty:
        df = df.sort_values("Margin_dB", ascending=True)
    return df