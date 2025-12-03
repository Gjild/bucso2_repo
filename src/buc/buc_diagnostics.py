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

    UPDATED to be consistent with the engine's intra-tile IF1 sweep:

    - IF1 is swept across the tile bandwidth using engine.tile_if1_sweep_step_hz.
    - For each IF1 tone, we compute its own desired IF2 and RF based on LO1/LO2 and
      chain sense (Sum-Sum vs Diff-Diff).
    - Desired-path tones for each swept IF1 are suppressed from the spur ledger:
        * Stage-1: (m,n) = (1,1) term that lands near that IF1's desired IF2.
        * Stage-2: (m,n) = (1,1) term that lands near that IF1's desired RF.
      These are treated as the wanted carriers for that IF1, not spurs.
    """

    tile = next(t for t in engine.cfg.tiles if t.id == tile_id)
    lo1_freq = policy_row["lo1_hz"]
    lo2_freq = policy_row["lo2_hz"]
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
    corr1 = max(
        -engine.hw.mixer1.scaling_cap, min(engine.hw.mixer1.scaling_cap, corr1)
    )

    corr2 = (p2 - engine.hw.mixer2.nom_drive_dbm) * engine.hw.mixer2.scaling_slope
    corr2 = max(
        -engine.hw.mixer2.scaling_cap, min(engine.hw.mixer2.scaling_cap, corr2)
    )

    side_str = policy_row.get("lo1_side", "low")
    is_sum_mix = side_str == "low"
    high_side = not is_sum_mix

    # IF1 harmonic model, kept consistent with the engine.
    if1_harmonics = getattr(engine.cfg, "if1_harmonics", None)
    if not if1_harmonics:
        if1_harmonics = [(1, 0.0)]

    # High-level chain sense + desired path formulas (symbolic)
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
        if1_hz: float | None = None,
        if1_harm_k: int | None = None,
    ):
        """
        Helper to append a row with full context.
        """
        guard = engine.cfg.yaml_data["constraints"]["guard_margin_db"]
        margin = limit_db - level_db - guard

        # Combined spur "order" across stages, when known
        total_order = None
        if stage1_m is not None and stage1_n is not None:
            total_order = abs(stage1_m) + abs(stage1_n)
        if stage2_m is not None and stage2_n is not None:
            stage2_order = abs(stage2_m) + abs(stage2_n)
            total_order = (total_order or 0) + stage2_order

        ledger.append(
            {
                # --- Tile & filter context ---
                "Tile_ID": tile.id,
                "IF1_center_GHz": tile.if1_center_hz / 1e9,
                "BW_MHz": tile.bw_hz / 1e6,
                "RF_center_GHz": tile.rf_center_hz / 1e9,
                "IF2_center_GHz": if2_filter.center_hz / 1e9,
                "IF2_bw_MHz": if2_filter.bw_hz / 1e6,

                # Extra sweep context
                "IF1_Hz": if1_hz,
                "IF1_GHz": None if if1_hz is None else if1_hz / 1e9,
                "IF1_harm_k": if1_harm_k,

                # --- LO context ---
                "LO1_GHz": lo1_freq / 1e9,
                "LO2_GHz": lo2_freq / 1e9,

                # --- Desired-path context ---
                "Chain_Sense": chain_sense,  # "Sum-Sum" or "Diff-Diff"
                "Desired_S1_Formula": desired_s1_formula,  # IF1 -> IF2 desired mapping
                "Desired_S2_Formula": desired_s2_formula,  # IF2 -> RF desired mapping

                # --- Spur identification ---
                "Source_Stage": source_stage,
                "Stage": stage_lbl,  # 2 or "1+2"
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
            }
        )

    # ------------------------------------------------------------------
    # IF1 sweep setup
    # ------------------------------------------------------------------
    sweep_step = float(engine.tile_if1_sweep_step_hz)
    if sweep_step <= 0.0:
        # Old behaviour: single IF1 tone at tile centre
        if1_tones = [tile.if1_center_hz]
    else:
        if_bw_half = 0.5 * tile.bw_hz
        f_start = tile.if1_center_hz - if_bw_half
        f_stop = tile.if1_center_hz + if_bw_half
        if sweep_step < 0.0:
            sweep_step = -sweep_step
        if sweep_step == 0.0:
            sweep_step = tile.bw_hz
        if1_tones = []
        f_if1 = f_start
        while f_if1 <= f_stop + 1e-9:
            if1_tones.append(f_if1)
            f_if1 += sweep_step

    # IF2 desired tolerance for desired-path detection (per IF1)
    tol_if2 = max(100.0, engine.grid_step)

    # ------------------------------------------------------------------
    # Stage-1 spur collection across IF1 sweep
    # We collect (f_if2, lvl_after_if2, m1, n1, src_tag, if1_hz, k_if)
    # Also keep a copy *before* IF2 filter for possible diagnostics (not needed for S2).
    # ------------------------------------------------------------------
    stage1_spurs = []

    # First, LO-only families (n == 0) – independent of IF1 tone and its harmonics.
    # These *do not* depend on sweep IF1 frequency, so we can do them once.
    lo_only_spurs = []
    for k in range(len(lo1_spec)):
        f_lo_c, p_lo_c = lo1_spec[k]
        lo_tag = "LO1" if k == 0 else "LO1H"

        for row in engine.hw.mixer1.spur_table_np:
            m, n, base_rej = int(row[0]), int(row[1]), row[2]
            rej = base_rej - corr1
            lvl = p_lo_c + rej

            if n != 0:
                continue  # not LO-only

            # Pure LO tone at m * f_LO1
            f_spur_if2 = abs(m * f_lo_c)
            atten_if2 = get_lut_val(
                f_spur_if2, engine.if2_lut_buffer, engine.grid_step
            )
            lvl_input = lvl - atten_if2
            if lvl_input < noise_floor:
                continue

            formula = f"({m}*{lo_tag})"
            lo_only_spurs.append(
                (f_spur_if2, lvl_input, formula, m, n, "LO-only")
            )

    # Now IF-dependent families (n > 0) for each IF1 harmonic and each swept IF1 tone
    for if1_hz in if1_tones:
        # Per-IF1 desired IF2 frequency
        if engine.cfg.enforce_desired_mn11_only:
            if high_side:
                # Diff-Diff desired: |LO1 - IF1|
                f_if2_desired = abs(lo1_freq - if1_hz)
            else:
                # Sum-Sum desired: LO1 + IF1
                f_if2_desired = lo1_freq + if1_hz
        else:
            f_if2_desired = if2_filter.center_hz

        # Add LO-only spurs for this IF1 (they don't depend on IF1, but we tag IF1 in the ledger)
        for f_spur_if2, lvl_input, formula, m, n, src_tag in lo_only_spurs:
            # No desired-path suppression here: n == 0 cannot be desired (1,1)
            stage1_spurs.append(
                (f_spur_if2, lvl_input, formula, m, n, src_tag, if1_hz, None)
            )

        # IF-dependent terms
        for k_if, rel_if_dbc in if1_harmonics:
            if1_freq = k_if * if1_hz

            for k in range(len(lo1_spec)):
                f_lo_c, p_lo_c = lo1_spec[k]
                is_main = k == 0
                lo_tag = "LO1" if k == 0 else "LO1H"

                for row in engine.hw.mixer1.spur_table_np:
                    m, n, base_rej = int(row[0]), int(row[1]), row[2]
                    if n == 0:
                        continue  # LO-only; already handled

                    rej = base_rej - corr1
                    lvl = p_lo_c + rej  # base mixer spur level

                    eff_lo = m * f_lo_c

                    # PURE IF BUGFIX: avoid double-counting for m == 0, n != 0.
                    pure_if_family = m == 0 and n != 0
                    s_if_values = (1,) if pure_if_family else (-1, 1)

                    for s_if in s_if_values:
                        # Desired-path detection only for the fundamental harmonic,
                        # main LO component and (m,n) = (1,1), and for THIS IF1 tone.
                        is_desired = False
                        if engine.cfg.enforce_desired_mn11_only:
                            if k_if == 1 and is_main and m == 1 and n == 1:
                                f_candidate = abs(eff_lo + s_if * n * if1_freq)
                                if abs(f_candidate - f_if2_desired) < tol_if2:
                                    is_desired = True

                        if is_desired:
                            # This is the desired IF2 for this IF1 tone; skip from spur list.
                            continue

                        f_spur_if2 = abs(eff_lo + s_if * n * if1_freq)
                        atten_if2 = get_lut_val(
                            f_spur_if2, engine.if2_lut_buffer, engine.grid_step
                        )

                        # Apply harmonic amplitude offset rel_if_dbc (≤ 0 dB typical)
                        lvl_input = (lvl + rel_if_dbc) - atten_if2
                        if lvl_input < noise_floor:
                            continue

                        sign_if = "+" if s_if > 0 else "-"
                        formula = f"({m}*{lo_tag} {sign_if} {n}*IF1_k={k_if})"

                        source_tag = f"IF1 k={k_if}"
                        stage1_spurs.append(
                            (
                                f_spur_if2,
                                lvl_input,
                                formula,
                                m,
                                n,
                                source_tag,
                                if1_hz,
                                k_if,
                            )
                        )

    # ------------------------------------------------------------------
    # B. Direct Stage-2 spurs from desired IF2 tones (per IF1 tone)
    # ------------------------------------------------------------------
    for if1_hz in if1_tones:
        # Desired IF2 & RF for this IF1
        if engine.cfg.enforce_desired_mn11_only:
            if high_side:
                f_if2_desired = abs(lo1_freq - if1_hz)
                rf_desired = abs(lo2_freq - f_if2_desired)
            else:
                f_if2_desired = lo1_freq + if1_hz
                rf_desired = lo2_freq + f_if2_desired
        else:
            f_if2_desired = if2_filter.center_hz
            if high_side:
                rf_desired = abs(lo2_freq - f_if2_desired)
            else:
                rf_desired = lo2_freq + f_if2_desired

        tol_rf = max(100.0, engine.grid_step)

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
                    # This pure-LO spur is independent of IF1 tone; keep per IF1 for consistency
                    atten_rf = get_lut_val(
                        f_spur_rf, engine.rf_lut, engine.grid_step
                    )
                    final_lvl = lvl - atten_rf
                    if final_lvl < noise_floor:
                        continue

                    limit = get_lut_val(
                        f_spur_rf, engine.mask_lut, engine.grid_step
                    )
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
                            if1_hz=if1_hz,
                            if1_harm_k=None,
                        )
                    continue

                # n > 0 : ±IF2 products (sum/diff), LO sign removed
                eff_lo = m * f_lo_c  # +m * LO2

                # PURE-IF BUGFIX (Stage-2) – see kernels
                pure_if_family = m == 0 and n != 0
                s_if_values = (1,) if pure_if_family else (-1, 1)

                for s_if in s_if_values:
                    f_spur_rf = abs(eff_lo + s_if * n * f_if2_desired)

                    # Filter out the desired RF carrier for THIS IF1:
                    # main LO, (m,n) = (1,1), and near rf_desired.
                    if (
                        engine.cfg.enforce_desired_mn11_only
                        and k == 0
                        and m == 1
                        and n == 1
                        and abs(f_spur_rf - rf_desired) < tol_rf
                    ):
                        continue

                    atten_rf = get_lut_val(
                        f_spur_rf, engine.rf_lut, engine.grid_step
                    )
                    final_lvl = lvl - atten_rf
                    if final_lvl < noise_floor:
                        continue

                    limit = get_lut_val(
                        f_spur_rf, engine.mask_lut, engine.grid_step
                    )
                    if (limit - final_lvl) < report_threshold_db:
                        sign_if = "+" if s_if > 0 else "-"
                        formula = f"({m}*{lo_tag} {sign_if} {n}*IF2_desired)"
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
                            if1_hz=if1_hz,
                            if1_harm_k=None,
                        )

    # ------------------------------------------------------------------
    # C. Stage-1 leakage spurs mixed again in Stage-2
    # ------------------------------------------------------------------
    for (
        sp_f,
        sp_l,
        sp_form,
        m1,
        n1,
        src_tag,
        if1_hz,
        k_if,
    ) in stage1_spurs:
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
                    atten_rf = get_lut_val(
                        f_final_rf, engine.rf_lut, engine.grid_step
                    )
                    lvl_rf = l_eff + lvl_stage2
                    final_lvl = lvl_rf - atten_rf
                    if final_lvl < noise_floor:
                        continue

                    limit = get_lut_val(
                        f_final_rf, engine.mask_lut, engine.grid_step
                    )
                    if (limit - final_lvl) < report_threshold_db:
                        formula = f"{m}*{lo_tag}"
                        add_entry(
                            f"Leakage (S1->S2, {src_tag})",
                            "1+2",
                            formula,
                            f_final_rf,
                            final_lvl,
                            limit,
                            stage1_m=m1,
                            stage1_n=n1,
                            stage2_m=m,
                            stage2_n=n,
                            if1_hz=if1_hz,
                            if1_harm_k=k_if,
                        )
                    continue

                # n > 0 : two sum/diff products relative to the Stage-1 spur
                for s_if in (-1, 1):
                    f_final_rf = abs(eff_lo + s_if * n * sp_f)

                    lvl_rf = l_eff + lvl_stage2  # S1 level (dB) + S2 gain (dB)
                    atten_rf = get_lut_val(
                        f_final_rf, engine.rf_lut, engine.grid_step
                    )
                    final_lvl = lvl_rf - atten_rf

                    if final_lvl < noise_floor:
                        continue

                    limit = get_lut_val(
                        f_final_rf, engine.mask_lut, engine.grid_step
                    )

                    if (limit - final_lvl) < report_threshold_db:
                        clean_sp_form = (
                            sp_form.replace("(", "").replace(")", "")
                        )
                        sign_if = "+" if s_if > 0 else "-"
                        formula = (
                            f"{m}*{lo_tag} {sign_if} {n}*[{clean_sp_form}]"
                        )
                        add_entry(
                            f"Leakage (S1->S2, {src_tag})",
                            "1+2",
                            formula,
                            f_final_rf,
                            final_lvl,
                            limit,
                            stage1_m=m1,
                            stage1_n=n1,
                            stage2_m=m,
                            stage2_n=n,
                            if1_hz=if1_hz,
                            if1_harm_k=k_if,
                        )

    # ------------------------------------------------------------------
    # D. Mixer-2 IF→RF isolation as a direct RF-port spur
    # ------------------------------------------------------------------
    if engine.hw.mixer2.include_isolation_spurs:
        f_if2_leak = float(if2_filter.center_hz)
        if f_if2_leak > 0.0:
            # RF filter attenuation at IF2-like frequency
            atten_rf = get_lut_val(
                f_if2_leak, engine.rf_lut, engine.grid_step
            )
            iso_db = engine.hw.mixer2.if_feedthrough_rej_db()  # e.g. -45 dBc
            final_lvl = iso_db - atten_rf

            if final_lvl >= noise_floor:
                limit = get_lut_val(
                    f_if2_leak, engine.mask_lut, engine.grid_step
                )

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
                        if1_hz=None,
                        if1_harm_k=None,
                    )

    df = pd.DataFrame(ledger)
    if not df.empty:
        # Expanded column set; old columns are preserved so downstream
        # tooling that expects them still works.
        cols = [
            # Context
            "Tile_ID",
            "IF1_center_GHz",
            "BW_MHz",
            "RF_center_GHz",
            "IF2_center_GHz",
            "IF2_bw_MHz",
            "LO1_GHz",
            "LO2_GHz",
            "Chain_Sense",
            "Desired_S1_Formula",
            "Desired_S2_Formula",
            # Sweep context (new)
            "IF1_Hz",
            "IF1_GHz",
            "IF1_harm_k",
            # Spur identification
            "Source_Stage",
            "Stage",
            "Stage1_m",
            "Stage1_n",
            "Stage2_m",
            "Stage2_n",
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
