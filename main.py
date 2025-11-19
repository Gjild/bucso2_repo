import time
import pandas as pd
import numpy as np
from colorama import Fore, Style
from multiprocessing import Pool, cpu_count

from buc_structures import GlobalConfig, FilterModel
from buc_models import HardwareStack
from buc_engine import SpurEngine
from buc_diagnostics import generate_spur_ledger
from buc_visuals import plot_margin_heatmap

# Global engine instance for workers
worker_engine = None

def init_worker(config, stack_def):
    global worker_engine
    hw = HardwareStack(config, stack_def)
    worker_engine = SpurEngine(config, hw)

def evaluate_wrapper(if2_model):
    # Returns (margin, retune_cost, model)
    score, retunes = worker_engine.evaluate_policy(if2_model, search_mode=True)
    return (score, retunes, if2_model)

def local_refinement(engine, seed_if2, initial_score, initial_retunes):
    """ Performs coordinate descent around the seed (Center AND BW). """
    current_if2 = seed_if2
    current_score = initial_score
    current_retunes = initial_retunes
    
    # Coarse then Fine steps
    freq_steps = [50e6, 10e6, 2e6]
    # Added Bandwidth refinement steps
    bw_steps = [20e6, 5e6] 
    
    print(f"{Fore.MAGENTA}--- Starting Local Refinement on Seed {seed_if2.center_hz/1e9:.2f}GHz / BW {seed_if2.bw_hz/1e6}MHz ---{Style.RESET_ALL}")
    
    # 1. Refine Center Frequency
    for step in freq_steps:
        improved = True
        while improved:
            improved = False
            neighbors = []
            base_c = current_if2.center_hz
            
            for d in [-step, step]:
                neighbors.append(FilterModel(
                    base_c + d, current_if2.bw_hz, 
                    current_if2.model_type, current_if2.passband_il, 
                    current_if2.rolloff, current_if2.stop_floor
                ))
            
            for n in neighbors:
                s, r = engine.evaluate_policy(n, search_mode=True)
                
                # Improvement criteria: Better margin OR (Similar margin AND fewer retunes)
                # E.g. Margin gain > 0.01 dB is strictly better
                # If Margin within 0.1 dB, prefer fewer retunes
                is_better = False
                if s > current_score + 0.01:
                    is_better = True
                elif abs(s - current_score) < 0.1 and r < current_retunes:
                    is_better = True
                
                if is_better:
                    current_score = s
                    current_retunes = r
                    current_if2 = n
                    improved = True
                    print(f"  > Improved Freq: {current_if2.center_hz/1e9:.3f}GHz -> {current_score:.2f} dB (Retunes: {r})")

    # 2. Refine Bandwidth
    for step in bw_steps:
        improved = True
        while improved:
            improved = False
            neighbors = []
            base_bw = current_if2.bw_hz
            
            for d in [-step, step]:
                # Enforce min/max BW logic if needed, though physics will naturally penalize
                new_bw = base_bw + d
                if new_bw < 100e6: continue 
                neighbors.append(FilterModel(
                    current_if2.center_hz, new_bw, 
                    current_if2.model_type, current_if2.passband_il, 
                    current_if2.rolloff, current_if2.stop_floor
                ))
                
            for n in neighbors:
                s, r = engine.evaluate_policy(n, search_mode=True)
                
                is_better = False
                if s > current_score + 0.01:
                    is_better = True
                elif abs(s - current_score) < 0.1 and r < current_retunes:
                    is_better = True
                    
                if is_better:
                    current_score = s
                    current_retunes = r
                    current_if2 = n
                    improved = True
                    print(f"  > Improved BW: {current_if2.bw_hz/1e6:.1f}MHz -> {current_score:.2f} dB")

    # Final verification with full physics
    final_score, final_retunes = engine.evaluate_policy(current_if2, search_mode=False)
    return current_if2, final_score, final_retunes

def build_policy_csv(engine, if2_filter, tiles):
    rows = []
    for t in tiles:
        # Note: get_all_valid_candidates runs full physics (search_mode=False)
        cands = engine.get_all_valid_candidates(t, if2_filter, min_margin_db=engine.opt_cutoff)
        
        if not cands:
            rows.append({"tile_id": t.id, "margin": -999, "if1_hz": t.if1_center_hz, "rf_hz": t.rf_center_hz})
            continue
            
        # We take the best candidate (index 0). 
        # NOTE: A full production tool would re-run the Hysteresis Logic here 
        # across tiles to select the specific LOs. For now, we take the best margin result.
        best = cands[0]
        margin, lo1, lo2 = best
        
        rows.append({
            "tile_id": t.id,
            "if1_hz": t.if1_center_hz,
            "rf_hz": t.rf_center_hz,
            "lo1_hz": lo1.freq_hz, "lo1_pad": lo1.pad_db, "lo1_side": lo1.side,
            "lo2_hz": lo2.freq_hz, "lo2_pad": lo2.pad_db, "lo2_side": lo2.side,
            "margin": margin,
            "if2_center": if2_filter.center_hz,
            "if2_bw": if2_filter.bw_hz
        })
    return pd.DataFrame(rows)

def main():
    print(f"{Fore.CYAN}=== Dual-Conversion BUC Spur Optimizer (Accelerated) ==={Style.RESET_ALL}")
    cfg = GlobalConfig.load("config.yaml")
    stacks = cfg.yaml_data['hardware_choices']['stacks']
    
    best_global_score = -999
    best_global_setup = None 
    
    num_workers = max(1, cpu_count() - 1)
    TOP_K_SEEDS = cfg.yaml_data['if2_model']['search'].get('top_k_seeds_for_refinement', 3)
    
    for stack_def in stacks:
        print(f"\nEvaluating Hardware Stack: {Fore.GREEN}{stack_def['name']}{Style.RESET_ALL}")
        if2_cfg = cfg.yaml_data['if2_model']
        
        ftype = 1 if (if2_cfg.get('scaled_s2p', {}).get('enabled', False)) else 0
        
        centers = np.arange(if2_cfg['center_range_hz'][0], if2_cfg['center_range_hz'][1], if2_cfg['search']['coarse_center_step_hz'])
        bws = np.arange(if2_cfg['min_bw_hz'], if2_cfg['max_bw_hz'], if2_cfg['search']['coarse_bw_step_hz'])
        
        tasks = []
        for c in centers:
            for bw in bws:
                tasks.append(FilterModel(
                    c, bw, ftype, 
                    if2_cfg['passband_il_db'], 
                    if2_cfg['rolloff_db_per_dec'], 
                    if2_cfg['stop_floor_db']
                ))
        
        print(f"  Scanning {len(tasks)} coarse points with {num_workers} workers...")
        
        with Pool(num_workers, initializer=init_worker, initargs=(cfg, stack_def)) as p:
            results = p.map(evaluate_wrapper, tasks)
        
        # Sort by Margin (Desc) first, then Retunes (Asc)
        # Tuple comparison: (Margin, -Retunes) to optimize both using default sort?
        # Python sorts tuples element-wise. We want Max Margin, Min Retunes.
        # results contains (margin, retunes, model).
        # Sort key: (margin, -retunes)
        results.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        
        top_seeds = results[:TOP_K_SEEDS]
        print(f"  Top {len(top_seeds)} Coarse Seeds selected for refinement.")
        
        hw = HardwareStack(cfg, stack_def)
        engine = SpurEngine(cfg, hw)
        
        for seed_score, seed_retunes, seed_model in top_seeds:
            if seed_score <= -900: continue 
            
            refined_model, refined_score, refined_retunes = local_refinement(engine, seed_model, seed_score, seed_retunes)
            
            if refined_score > best_global_score:
                best_global_score = refined_score
                best_global_setup = (stack_def, refined_model)
                print(f"  {Fore.YELLOW}New Global Best: {best_global_score:.2f} dB (Retunes: {refined_retunes}){Style.RESET_ALL}")

    if not best_global_setup:
        print("No solution found.")
        return

    stack_def, final_if2 = best_global_setup
    hw = HardwareStack(cfg, stack_def)
    engine = SpurEngine(cfg, hw)
    
    print(f"\n{Fore.YELLOW}Final Selected Config: Margin {best_global_score:.2f} dB{Style.RESET_ALL}")
    print(f"Stack: {stack_def['name']}")
    print(f"IF2 Center: {final_if2.center_hz/1e9:.4f} GHz, BW: {final_if2.bw_hz/1e6:.1f} MHz")
    
    # 2. Generate Policy
    df_policy = build_policy_csv(engine, final_if2, cfg.tiles)
    df_policy.to_csv("final_policy.csv", index=False)
    print("Policy saved to final_policy.csv")
    
    # 3. Visualization
    plot_margin_heatmap(df_policy)
    
    # 4. Diagnostics
    valid_rows = df_policy[df_policy['margin'] > -900]
    if not valid_rows.empty:
        worst_row = valid_rows.loc[valid_rows['margin'].idxmin()]
        print(f"\nRunning Diagnostics for Worst Case: Tile {worst_row['tile_id']} (Margin {worst_row['margin']:.2f} dB)")
        
        engine.evaluate_policy(final_if2, search_mode=False)

        ledger = generate_spur_ledger(engine, int(worst_row['tile_id']), worst_row, final_if2)
        if not ledger.empty:
            print(ledger.head(10).to_string(index=False))
            ledger.to_csv("worst_case_ledger.csv", index=False)
            print(f"Detailed ledger saved to worst_case_ledger.csv")

if __name__ == "__main__":
    main()