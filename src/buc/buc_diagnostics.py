import pandas as pd
import numpy as np
from .buc_structures import Tile, FilterModel
from .buc_kernels import get_lut_val


def generate_spur_ledger(
    engine,
    tile_id,
    policy_row,
    if2_filter: FilterModel,
    report_threshold_db: float = 10.0,
):
    """
    Build a detailed spur ledger for a single tile / policy setting.
    """
    tile = next(t for t in engine.cfg.tiles if t.id == tile_id)
    lo1_freq = policy_row['lo1_hz']
    lo2_freq = policy_row['lo2_hz']
    noise_floor = engine.noise_floor
    max_spur_dbc = engine.cfg.max_spur_level_dbc

    # Regenerate raw spectra
    lo1_spec = engine.hw.generate_lo_spectrum(engine.hw.lo1_def, lo1_freq)
    lo2_spec = engine.hw.generate_lo_spectrum(engine.hw.lo2_def, lo2_freq)

    _, _, p1 = engine.hw.get_valid_lo_config(
        engine.hw.lo1_def, lo1_freq, engine.hw.mixer1.drive_req
    )
    _, _, p2 = engine.hw.get_valid_lo_config(
        engine.hw.lo2_def, lo2_freq, engine.hw.mixer2.drive_req
    )

    corr1 = (p1 - engine.hw.mixer1.nom_drive_dbm) * engine.hw.mixer1.scaling_slope
    corr1 = max(-engine.hw.mixer1.scaling_cap,
                min(engine.hw.mixer1.scaling_cap, corr1))

    corr2 = (p2 - engine.hw.mixer2.nom_drive_dbm) * engine.hw.mixer2.scaling_slope
    corr2 = max(-engine.hw.mixer2.scaling_cap,
                min(engine.hw.mixer2.scaling_cap, corr2))

    side_str = policy_row.get('lo1_side', 'low')
    is_sum_mix = (side_str == 'low')

    # High-level chain sense + desired path formulas
    chain_sense = "Sum-Sum" if is_sum_mix else "Diff-Diff"
    if is_sum_mix:
        desired_s1_formula = "f_IF2_desired = f_LO1 + f_IF1"
        desired_s2_formula = "f_RF_desired  = f_LO2 + f_IF2"
    else:
        desired_s1_formula = "f_IF2_desired = |f_LO1 - f_IF1|"
        desired_s2_formula = "f_RF_desired  = |f_LO2 - f_IF2|"

    ledger = []

    def add_entry(
        source_stage: str,
        stage_lbl,
        formula: str,
        freq_hz: float,
        level_db: float,
        limit_db: float,
        stage1_m: float | None = None,
        stage1_n: float | None = None,
        stage2_m: float | None = None,
        stage2_n: float | None = None,
    ):
        """
        Helper to append a row with full context.
        """
        margin = limit_db - level_db - engine.cfg.yaml_data['constraints']['guard_margin_db']

        # Combined spur "order" across stages, when known
        total_order = None
        if stage1_m is not None and stage1_n is not None:
            total_order = abs(stage1_m) + abs(stage1_n)
        if stage2_m is not None and stage2_n is not None:
            stage2_order = abs(stage2_m) + abs(stage2_n)
            total_order = (total_order or 0) + stage2_order

        ledger.append({
            # --- Tile & filter context ---
            "Tile_ID": tile.id,
            "IF1_center_GHz": tile.if1_center_hz / 1e9,
            "BW_MHz": tile.bw_hz / 1e6,
            "RF_center_GHz": tile.rf_center_hz / 1e9,
            "IF2_center_GHz": if2_filter.center_hz / 1e9,
            "IF2_bw_MHz": if2_filter.bw_hz / 1e6,

            # --- LO context ---
            "LO1_GHz": lo1_freq / 1e9,
            "LO2_GHz": lo2_freq / 1e9,

            # --- Desired-path context ---
            "Chain_Sense": chain_sense,                # "Sum-Sum" or "Diff-Diff"
            "Desired_S1_Formula": desired_s1_formula,  # IF1 -> IF2 desired mapping
            "Desired_S2_Formula": desired_s2_formula,  # IF2 -> RF desired mapping

            # --- Spur identification ---
            "Source_Stage": source_stage,
            "Stage": stage_lbl,         # 2 or "1+2"
            "Stage1_m": stage1_m,
            "Stage1_n": stage1_n,
            "Stage2_m": stage2_m,
            "Stage2_n": stage2_n,
            "Total_Order": total_order,

            # --- Numeric results ---
            "Freq_GHz": freq_hz / 1e9,
            "Level_dBm_equiv": level_db,
            "Limit_dBc": limit_db,
            "Margin_dB": margin,

            # --- Human-readable spur formula ---
            "Formula": formula,
        })

    # Desired IF2 tone and tolerance for desired-path detection
    f_if2_desired = if2_filter.center_hz
    tol_if2 = max(100.0, engine.grid_step)  # same spirit as engine: small but >0

    stage1_spurs = []  # (f_if2, level_at_if2, formula, m1, n1)

    # ------------------------------------------------------------------
    # A. Stage-1 spurs at IF2 (used as input to Stage-2 leakage paths)
    # ------------------------------------------------------------------
    for k in range(len(lo1_spec)):
        f_lo_c, p_lo_c = lo1_spec[k]
        is_main = (k == 0)
        lo_tag = "LO1" if k == 0 else "LO1H"

        for row in engine.hw.mixer1.spur_table_np:
            m, n, base_rej = int(row[0]), int(row[1]), row[2]
            rej = base_rej - corr1
            lvl = p_lo_c + rej  # same as engine: LO component + scaled rej

            # n == 0 : pure LO term → one spur, no sum/diff over IF
            if n == 0:
                f_spur_if2 = abs(m * f_lo_c)
                atten_if2 = get_lut_val(f_spur_if2, engine.if2_lut_buffer, engine.grid_step)
                lvl_input = lvl - atten_if2
                if lvl_input < noise_floor:
                    continue

                formula = f"({m}*{lo_tag})"
                stage1_spurs.append((f_spur_if2, lvl_input, formula, m, n))
                continue

            # n > 0 : ±n recipes relative to IF1, LO sign removed
            eff_lo = m * f_lo_c  # +m * LO1

            # --- PURE-IF BUGFIX HERE ------------------------------------
            # For m == 0, f_spur = |0 ± n*f_IF1| = |n|*f_IF1, so +n and -n
            # land on the same frequency. Only generate ONE sign to avoid
            # 2× power double-counting.
            pure_if_family = (m == 0 and n != 0)
            s_if_values = (1,) if pure_if_family else (-1, 1)
            # -------------------------------------------------------------

            for s_if in s_if_values:
                # Identify desired path *by IF2 frequency* (like the engine):
                # Any main-LO (m,n)=(1,1) tone that lands at f_if2_desired
                # is considered the desired IF2 and should not be added
                # to the spur list.
                is_desired = False
                if is_main and m == 1 and n == 1:
                    f_candidate = abs(eff_lo + s_if * n * tile.if1_center_hz)
                    if abs(f_candidate - f_if2_desired) < tol_if2:
                        is_desired = True

                if is_desired:
                    continue

                f_spur_if2 = abs(eff_lo + s_if * n * tile.if1_center_hz)
                atten_if2 = get_lut_val(f_spur_if2, engine.if2_lut_buffer, engine.grid_step)
                lvl_input = lvl - atten_if2
                if lvl_input < noise_floor:
                    continue

                sign_if = "+" if s_if > 0 else "-"
                formula = f"({m}*{lo_tag} {sign_if} {n}*IF1)"

                # Keep stage-1 orders (m1, n1) for later leakage context
                stage1_spurs.append((f_spur_if2, lvl_input, formula, m, n))

    # IF feedthrough into Stage-1 list (for S2 processing)
    #if engine.hw.mixer2.include_isolation_spurs:
    #    rej_if = engine.hw.mixer2.if_feedthrough_rej_db()
    #    # Treat as a special "0,1" term (pure IF leaking through)
    #    stage1_spurs.append(
    #        (if2_filter.center_hz, rej_if, "(IF Feedthrough)", 0, 1)
    #    )

    # ------------------------------------------------------------------
    # B. Direct Stage-2 spurs from desired IF2 tone
    # ------------------------------------------------------------------
    for k in range(len(lo2_spec)):
        f_lo_c, p_lo_c = lo2_spec[k]
        lo_tag = "LO2" if k == 0 else "LO2H"

        for row in engine.hw.mixer2.spur_table_np:
            m, n, base_rej = int(row[0]), int(row[1]), row[2]
            rej = base_rej - corr2
            lvl = p_lo_c + rej  # LO comp + scaled rejection

            # n == 0 : pure LO term (no dependence on IF2)
            if n == 0:
                f_spur_rf = abs(m * f_lo_c)
                atten_rf = get_lut_val(f_spur_rf, engine.rf_lut, engine.grid_step)
                final_lvl = lvl - atten_rf
                if final_lvl < noise_floor:
                    continue

                limit = get_lut_val(f_spur_rf, engine.mask_lut, engine.grid_step)
                if (limit - final_lvl) < report_threshold_db:
                    formula = f"({m}*{lo_tag})"
                    add_entry(
                        "Direct (S2)",
                        2,
                        formula,
                        f_spur_rf,
                        final_lvl,
                        limit,
                        stage1_m=1,
                        stage1_n=1,
                        stage2_m=m,
                        stage2_n=n,
                    )
                continue

            # n > 0 : ±IF2 products (sum/diff), LO sign removed
            eff_lo = m * f_lo_c  # +m * LO2

            # --- PURE-IF BUGFIX HERE (Stage-2) --------------------------
            # Same reasoning as Stage-1: for m == 0, ±n collapses to a
            # pure-IF2 spur at |n|*f_IF2, so only generate one sign.
            pure_if_family = (m == 0 and n != 0)
            s_if_values = (1,) if pure_if_family else (-1, 1)
            # ------------------------------------------------------------

            for s_if in s_if_values:
                f_spur_rf = abs(eff_lo + s_if * n * f_if2_desired)

                # Filter out the *desired* RF carrier (main LO, m=1, n=1 @ RF center)
                if (k == 0 and m == 1 and n == 1 and
                        abs(f_spur_rf - tile.rf_center_hz) < 1000.0):
                    continue

                atten_rf = get_lut_val(f_spur_rf, engine.rf_lut, engine.grid_step)
                final_lvl = lvl - atten_rf
                if final_lvl < noise_floor:
                    continue

                limit = get_lut_val(f_spur_rf, engine.mask_lut, engine.grid_step)
                if (limit - final_lvl) < report_threshold_db:
                    sign_if = "+" if s_if > 0 else "-"
                    formula = f"({m}*{lo_tag} {sign_if} {n}*IF2_desired)"
                    # Stage-1 is the desired (1,1) path here; Stage-2 is (m,n)
                    add_entry(
                        "Direct (S2)",
                        2,
                        formula,
                        f_spur_rf,
                        final_lvl,
                        limit,
                        stage1_m=1,
                        stage1_n=1,
                        stage2_m=m,
                        stage2_n=n,
                    )

    # ------------------------------------------------------------------
    # C. Stage-1 leakage spurs mixed again in Stage-2
    # ------------------------------------------------------------------
    for sp_f, sp_l, sp_form, m1, n1 in stage1_spurs:
        # Apply clamp
        l_eff = sp_l
        if l_eff > max_spur_dbc:
            l_eff = max_spur_dbc

        for k in range(len(lo2_spec)):
            f_lo_c, p_lo_c = lo2_spec[k]
            lo_tag = "LO2" if k == 0 else "LO2H"

            for row in engine.hw.mixer2.spur_table_np:
                m, n, base_rej = int(row[0]), int(row[1]), row[2]
                rej = base_rej - corr2

                lvl_stage2 = p_lo_c + rej
                eff_lo = m * f_lo_c  # +m * LO2

                # n == 0 : pure LO spur; no dependence on sp_f
                if n == 0:
                    f_final_rf = abs(eff_lo)
                    atten_rf = get_lut_val(f_final_rf, engine.rf_lut, engine.grid_step)
                    lvl_rf = l_eff + lvl_stage2
                    final_lvl = lvl_rf - atten_rf
                    if final_lvl < noise_floor:
                        continue

                    limit = get_lut_val(f_final_rf, engine.mask_lut, engine.grid_step)
                    if (limit - final_lvl) < report_threshold_db:
                        formula = f"{m}*{lo_tag}"
                        add_entry(
                            "Leakage (S1->S2)",
                            "1+2",
                            formula,
                            f_final_rf,
                            final_lvl,
                            limit,
                            stage1_m=m1,
                            stage1_n=n1,
                            stage2_m=m,
                            stage2_n=n,
                        )
                    continue

                # n > 0 : two sum/diff products relative to the Stage-1 spur
                # (We intentionally keep ±n here, even for m=0; Stage-1
                #  has already de-duplicated pure-IF families so we won't
                #  hit the same 2× double-counting here.)
                for s_if in (-1, 1):
                    f_final_rf = abs(eff_lo + s_if * n * sp_f)

                    lvl_rf = l_eff + lvl_stage2  # S1 level (dB) + S2 gain (dB)
                    atten_rf = get_lut_val(f_final_rf, engine.rf_lut, engine.grid_step)
                    final_lvl = lvl_rf - atten_rf

                    if final_lvl < noise_floor:
                        continue

                    limit = get_lut_val(f_final_rf, engine.mask_lut, engine.grid_step)

                    if (limit - final_lvl) < report_threshold_db:
                        clean_sp_form = sp_form.replace("(", "").replace(")", "")
                        sign_if = "+" if s_if > 0 else "-"
                        formula = f"{m}*{lo_tag} {sign_if} {n}*[{clean_sp_form}]"
                        add_entry(
                            "Leakage (S1->S2)",
                            "1+2",
                            formula,
                            f_final_rf,
                            final_lvl,
                            limit,
                            stage1_m=m1,
                            stage1_n=n1,
                            stage2_m=m,
                            stage2_n=n,
                        )

    # ------------------------------------------------------------------
    # D. Mixer-2 IF→RF isolation as a direct RF-port spur
    # ------------------------------------------------------------------
    if engine.hw.mixer2.include_isolation_spurs:
        f_if2_leak = float(if2_filter.center_hz)
        if f_if2_leak > 0.0:
            # RF filter attenuation at IF2-like frequency
            atten_rf = get_lut_val(f_if2_leak, engine.rf_lut, engine.grid_step)
            iso_db = engine.hw.mixer2.if_feedthrough_rej_db()  # e.g. -45 dBc
            final_lvl = iso_db - atten_rf

            if final_lvl >= noise_floor:
                limit = get_lut_val(f_if2_leak, engine.mask_lut, engine.grid_step)

                # Only report if it is within the reporting window
                if (limit - final_lvl) < report_threshold_db:
                    add_entry(
                        source_stage="IF→RF Isolation (S2)",
                        stage_lbl=2,
                        formula="IF2_leakage (no mixing)",
                        freq_hz=f_if2_leak,
                        level_db=final_lvl,
                        limit_db=limit,
                        stage1_m=None,
                        stage1_n=None,
                        stage2_m=None,
                        stage2_n=None,
                    )

    df = pd.DataFrame(ledger)
    if not df.empty:
        # Expanded column set; old columns are preserved so downstream
        # tooling that expects them still works.
        cols = [
            # Context
            "Tile_ID",
            "IF1_center_GHz", "BW_MHz", "RF_center_GHz",
            "IF2_center_GHz", "IF2_bw_MHz",
            "LO1_GHz", "LO2_GHz",
            "Chain_Sense",
            "Desired_S1_Formula", "Desired_S2_Formula",

            # Spur identification
            "Source_Stage", "Stage",
            "Stage1_m", "Stage1_n",
            "Stage2_m", "Stage2_n",
            "Total_Order",

            # Numeric results (old core columns)
            "Freq_GHz",
            "Margin_dB",
            "Level_dBm_equiv",
            "Limit_dBc",

            # Human-readable spur expression
            "Formula",
        ]
        # Keep only the columns we defined (defensive in case of typos)
        cols = [c for c in cols if c in df.columns]
        df = df[cols].sort_values("Margin_dB", ascending=True)

    return df
