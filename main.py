import time
import pandas as pd
import numpy as np
import hashlib
import json
import platform
import sys
from pathlib import Path  # NEW
from colorama import Fore, Style
from multiprocessing import Pool, cpu_count

from buc import (
    GlobalConfig,
    FilterModel,
    HardwareStack,
    SpurEngine,
    generate_spur_ledger,
    plot_margin_heatmap,
    plot_if2_filter,
    basic_validation,
    markov_lock_summary,
)

worker_engine = None


def print_progress(current, total, prefix: str = "", length: int = 40, color=None):
    """
    Simple in-terminal progress bar.

    current : number of completed items
    total   : total items
    prefix  : text printed before the bar (e.g. '    ' or '  Scan ')
    length  : bar width in characters
    color   : optional colorama Fore.* color
    """
    if total <= 0:
        total = 1

    frac = current / total
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0

    filled = int(length * frac)
    bar = "=" * filled + "." * (length - filled)

    color_prefix = color or ""
    color_suffix = Style.RESET_ALL if color else ""

    sys.stdout.write(
        f"\r{prefix}{color_prefix}[{bar}] {100.0 * frac:5.1f}% ({current}/{total}){color_suffix}"
    )
    sys.stdout.flush()

    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def init_worker(config, stack_def):
    global worker_engine
    hw = HardwareStack(config, stack_def)
    worker_engine = SpurEngine(config, hw)


def evaluate_wrapper(if2_model):
    score, time_cost = worker_engine.evaluate_policy(if2_model, search_mode=True)
    return (score, time_cost, if2_model)


def local_refinement(engine, seed_if2, initial_score, initial_time):
    current_if2 = seed_if2
    current_score = initial_score
    current_time = initial_time

    if2_cfg = engine.cfg.yaml_data['if2_model']
    c_min, c_max = if2_cfg['center_range_hz']
    c_min = float(c_min)
    c_max = float(c_max)

    bw_min = float(if2_cfg['min_bw_hz'])
    bw_max = float(if2_cfg['max_bw_hz'])

    freq_steps = [50e6, 10e6, 2e6]
    bw_steps = [20e6, 5e6]

    print(f"{Fore.MAGENTA}--- Starting Local Refinement on Seed {seed_if2.center_hz/1e9:.2f}GHz ---{Style.RESET_ALL}")

    # Center refinement
    for step in freq_steps:
        improved = True
        while improved:
            improved = False
            neighbors = []
            base_c = current_if2.center_hz
            for d in [-step, step]:
                new_c = base_c + d
                if new_c < c_min or new_c > c_max:
                    continue
                neighbors.append(FilterModel(
                    new_c, current_if2.bw_hz,
                    current_if2.model_type, current_if2.passband_il,
                    current_if2.rolloff, current_if2.stop_floor
                ))
            for n in neighbors:
                s, t = engine.evaluate_policy(n, search_mode=True)
                is_better = False
                if s > current_score + 0.01:
                    is_better = True
                elif abs(s - current_score) < 0.1 and t < current_time:
                    is_better = True

                if is_better:
                    current_score, current_time, current_if2 = s, t, n
                    improved = True
                    print(f"  > Improved Freq: {current_if2.center_hz/1e9:.3f}GHz -> {current_score:.2f} dB")

    # Bandwidth refinement
    for step in bw_steps:
        improved = True
        while improved:
            improved = False
            neighbors = []
            base_bw = current_if2.bw_hz
            for d in [-step, step]:
                new_bw = base_bw + d
                if new_bw < bw_min or new_bw > bw_max:
                    continue
                neighbors.append(FilterModel(
                    current_if2.center_hz, new_bw,
                    current_if2.model_type, current_if2.passband_il,
                    current_if2.rolloff, current_if2.stop_floor
                ))
            for n in neighbors:
                s, t = engine.evaluate_policy(n, search_mode=True)
                is_better = False
                if s > current_score + 0.01:
                    is_better = True
                elif abs(s - current_score) < 0.1 and t < current_time:
                    is_better = True

                if is_better:
                    current_score, current_time, current_if2 = s, t, n
                    improved = True
                    print(f"  > Improved BW: {current_if2.bw_hz/1e6:.1f}MHz -> {current_score:.2f} dB")

    final_score, final_time = engine.evaluate_policy(current_if2, search_mode=False)
    return current_if2, final_score, final_time, getattr(engine, "last_retune_count", None)


def build_policy_df_from_entries(entries, if2_filter, hw_stack, cfg):
    rows = []
    for e in entries:
        lo1, lo2 = e.lo1, e.lo2
        side = e.side

        # Desired path tuples for this phase (fixed (1,1) family)
        s1_tup = "(1, -1)" if side == "high" else "(1, 1)"
        s2_tup = s1_tup

        rows.append({
            "tile_id": e.tile_id,
            "if1_center_hz": e.if1_center_hz,
            "bw_hz": e.bw_hz,
            "rf_center_hz": e.rf_center_hz,

            "lo1_hz": lo1.freq_hz,
            "lo1_mode": lo1.mode,
            "lo1_vco_div": 1,
            "lo1_drive_dbm": lo1.delivered_power_dbm,
            "lo1_pad_db": lo1.pad_db,
            "lo1_side": side,

            "lo2_hz": lo2.freq_hz,
            "lo2_mode": lo2.mode,
            "lo2_vco_div": 1,
            "lo2_drive_dbm": lo2.delivered_power_dbm,
            "lo2_pad_db": lo2.pad_db,
            "lo2_side": side,

            "spur_margin_db": e.spur_margin_db,
            "margin": e.spur_margin_db,  # deprecated alias

            "brittleness_db_per_step": e.brittleness_db_per_step,
            "lock_time_ms_tile": e.lock_time_ms_tile,

            "if2_center_hz": if2_filter.center_hz,
            "if2_bw_hz": if2_filter.bw_hz,
            "if2_model_type": (
                "symmetric_powerlaw" if if2_filter.model_type == 0
                else "scaled_s2p_prototype"
            ),

            "rf_bpf_id": getattr(hw_stack, "rf_bpf_id", "rf_bpf_synthetic"),

            "desired_stage1_tuple": s1_tup,
            "desired_stage2_tuple": s2_tup,
            "notes": "",
        })

    return pd.DataFrame(rows)


def sha256_file(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def export_metadata(cfg, best_global, out_dir: Path):
    if2 = best_global['if2_model']
    stack_def = best_global['stack_def']

    final_policy_path = out_dir / "final_policy.csv"

    meta = {
        "project": cfg.yaml_data.get("project", {}),
        "seed": cfg.yaml_data.get("project", {}).get("seed", None),
        "grid": {
            "grid_step_hz": cfg.grid_step_hz,
            "rbw_hz": cfg.rbw_hz,
            "grid_max_freq_hz": cfg.grid_max_freq_hz,
        },
        "orders": {
            "m1_max_order": cfg.m1_max_order,
            "m2_max_order": cfg.m2_max_order,
            "cross_stage_sum_max": cfg.cross_stage_sum_max,
        },
        "hardware": {
            "stack_name": stack_def.get("name"),
            "rf_bpf_file": stack_def.get("rf_bpf_file"),
        },
        "runtime": {
            "max_spur_level_dbc": cfg.max_spur_level_dbc,
            "dominant_prune_cutoff_db": cfg.dominant_prune_cutoff_db,
        },
        "files": {
            "config_yaml_sha256": sha256_file("config.yaml"),
            "final_policy_sha256": sha256_file(final_policy_path),
        },
        "if2_selected": {
            "center_hz": if2.center_hz,
            "bw_hz": if2.bw_hz,
        },
        "python": sys.version,
        "platform": platform.platform(),
    }

    meta_path = out_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Run metadata written to {meta_path}")


def main():
    print(f"{Fore.CYAN}=== Dual-Conversion BUC Spur Optimizer (Accelerated) ==={Style.RESET_ALL}")
    try:
        cfg = GlobalConfig.load("config.yaml")
        seed = cfg.yaml_data['project'].get('seed', 42)
        np.random.seed(seed)
    except Exception as e:
        print(f"{Fore.RED}Error loading configuration:{Style.RESET_ALL} {e}")
        return

    # NEW: Set up output directory at project root
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    print(f"Outputs will be written to: {out_dir}")

    stacks = cfg.yaml_data['hardware_choices']['stacks']
    all_global_results = []
    num_workers = max(1, cpu_count() - 1)
    TOP_K_SEEDS = cfg.yaml_data['if2_model']['search'].get('top_k_seeds_for_refinement', 3)

    for stack_def in stacks:
        print(f"\nEvaluating Hardware Stack: {Fore.GREEN}{stack_def['name']}{Style.RESET_ALL}")
        if2_cfg = cfg.yaml_data['if2_model']
        ftype = 1 if (if2_cfg.get('scaled_s2p', {}).get('enabled', False)) else 0

        # Force everything to float to avoid StrDType issues from YAML
        c_min, c_max = if2_cfg['center_range_hz']
        c_min = float(c_min)
        c_max = float(c_max)
        center_step = float(if2_cfg['search']['coarse_center_step_hz'])

        bw_min = float(if2_cfg['min_bw_hz'])
        bw_max = float(if2_cfg['max_bw_hz'])
        bw_step = float(if2_cfg['search']['coarse_bw_step_hz'])

        centers = np.arange(c_min, c_max, center_step, dtype=float)
        bws     = np.arange(bw_min, bw_max, bw_step, dtype=float)

        tasks = []
        for c in centers:
            for bw in bws:
                tasks.append(FilterModel(
                    c, bw, ftype, if2_cfg['passband_il_db'],
                    if2_cfg['rolloff_db_per_dec'], if2_cfg['stop_floor_db']
                ))

        total_tasks = len(tasks)
        print(f"  Scanning {total_tasks} coarse points with {num_workers} workers...")
        tasks.sort(key=lambda x: (x.center_hz, x.bw_hz))

        # Multiprocessing with live progress bar
        with Pool(num_workers, initializer=init_worker, initargs=(cfg, stack_def)) as p:
            results = []
            # Use imap_unordered so we get results as they complete
            for i, res in enumerate(p.imap_unordered(evaluate_wrapper, tasks), 1):
                results.append(res)

                # Throttle updates for huge grids: roughly 100 updates max
                if (
                    total_tasks <= 200
                    or i == total_tasks
                    or i % max(1, total_tasks // 100) == 0
                ):
                    print_progress(
                        i,
                        total_tasks,
                        prefix="    ",
                        length=40,
                        color=Fore.GREEN,
                    )

        #with Pool(num_workers, initializer=init_worker, initargs=(cfg, stack_def)) as p:
        #    results = p.map(evaluate_wrapper, tasks)

        results.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        top_seeds = results[:TOP_K_SEEDS]
        print(f"  Top {len(top_seeds)} Coarse Seeds selected for refinement.")

        hw = HardwareStack(cfg, stack_def)
        engine = SpurEngine(cfg, hw)

        for seed_score, seed_time, seed_model in top_seeds:
            if seed_score <= -900:
                continue
            ref_model, ref_score, ref_time, ref_retunes = local_refinement(engine, seed_model, seed_score, seed_time)
            all_global_results.append({
                "score": ref_score,
                "time": ref_time,
                "retunes": ref_retunes,
                "stack_def": stack_def,
                "if2_model": ref_model
            })
            print(f"  {Fore.YELLOW}Refined Result: {ref_score:.2f} dB (LockTime: {ref_time:.1f}ms, Retunes: {ref_retunes}){Style.RESET_ALL}")

    if not all_global_results:
        print("No solution found.")
        return

    all_global_results.sort(key=lambda x: (x['score'], -x['time']), reverse=True)

    unique_globals = []
    seen_centers = []
    for res in all_global_results:
        c_mhz = int(res['if2_model'].center_hz / 1e6)
        if not any(abs(c_mhz - s) < 50 for s in seen_centers):
            unique_globals.append(res)
            seen_centers.append(c_mhz)
        if len(unique_globals) >= 3:
            break

    # Export Alt Designs
    alt_rows = []
    for rank, res in enumerate(unique_globals, start=1):
        if2 = res['if2_model']
        alt_rows.append({
            "rank": rank,
            "stack_name": res['stack_def']['name'],
            "if2_center_hz": if2.center_hz,
            "if2_bw_hz": if2.bw_hz,
            "score_db": res['score'],
            "lock_time_ms": res['time'],
        })
    alt_path = out_dir / "alt_global_designs.csv"
    pd.DataFrame(alt_rows).to_csv(alt_path, index=False)
    print(f"Alternative global designs written to {alt_path}")

    best_global = unique_globals[0]
    print(f"\n{Fore.YELLOW}=== Final Selected Config ==={Style.RESET_ALL}")
    print(f"Score: {best_global['score']:.2f} dB, LockTime: {best_global['time']:.1f}ms")

    stack_def = best_global['stack_def']
    final_if2 = best_global['if2_model']
    hw = HardwareStack(cfg, stack_def)
    engine = SpurEngine(cfg, hw)

    # Re-build policy with no early exit and brittleness on
    print("Generating final policy...")
    _, _, _, entries = engine.build_policy_for_if2(
        final_if2,
        search_mode=False,
        compute_brittleness=True,
        stop_if_margin_below=None
    )

    df_policy = build_policy_df_from_entries(entries, final_if2, hw, cfg)
    policy_path = out_dir / "final_policy.csv"
    df_policy.to_csv(policy_path, index=False)
    print(f"Final policy written to {policy_path}")

    heatmap_path = out_dir / "heatmap_margin.png"
    plot_margin_heatmap(df_policy, filename=str(heatmap_path))
    
    # --- Export and plot final IF2 filter attenuation ---
    if2_csv_path = out_dir / "if2_filter_response.csv"
    if2_png_path = out_dir / "if2_filter_response.png"
    plot_if2_filter(engine, final_if2,
                    csv_path=str(if2_csv_path),
                    png_path=str(if2_png_path))

    if cfg.markov_matrix.size > 0:
        markov_path = out_dir / "markov_lock_summary.json"
        markov_lock_summary(df_policy, cfg.markov_matrix, out_path=str(markov_path))

    basic_validation(cfg, stack_def, final_if2)

    valid_rows = df_policy[df_policy['spur_margin_db'] > -900]
    if not valid_rows.empty:
        worst_row = valid_rows.loc[valid_rows['spur_margin_db'].idxmin()]
        print(f"\nRunning Diagnostics for Worst Case: Tile {worst_row['tile_id']} (Margin {worst_row['spur_margin_db']:.2f} dB)")

        ledger = generate_spur_ledger(engine, int(worst_row['tile_id']), worst_row, final_if2, report_threshold_db=1e9)
        if not ledger.empty:
            print(ledger.head(10).to_string(index=False))
            ledger_path = out_dir / "worst_case_ledger.csv"
            ledger.to_csv(ledger_path, index=False)
            print(f"Worst-case ledger written to {ledger_path}")

    export_metadata(cfg, best_global, out_dir)


if __name__ == "__main__":
    main()
