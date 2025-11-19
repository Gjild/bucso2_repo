import numpy as np
from buc_structures import Tile, FilterModel, LO_Candidate
from buc_kernels import (
    build_dense_lut, 
    fill_symmetric_filter_lut, 
    fill_scaled_s2p_lut,
    compute_stage1_intermediates,
    compute_stage2_from_intermediates
)

class SpurEngine:
    def __init__(self, config, hw_stack):
        self.cfg = config
        self.hw = hw_stack
        
        # 1. Build Global Grids and Static LUTs
        print("  Building Global LUTs...")
        self.grid_max = self.cfg.grid_max_freq_hz
        self.grid_step = self.cfg.grid_step_hz
        self.opt_cutoff = self.cfg.opt_cutoff_db
        self.noise_floor = self.cfg.noise_floor_dbc
        self.guard_db = float(self.cfg.yaml_data['constraints']['guard_margin_db'])
        
        # Mask LUT
        mask_oob_def = self.cfg.yaml_data['masks']['outofband']['default_dbc']
        self.mask_lut = np.full(int(self.grid_max/self.grid_step)+1, mask_oob_def, dtype=np.float32)
        
        ib_cfg = self.cfg.yaml_data['masks']['inband']
        rf_min = self.cfg.yaml_data['bands']['rf_hz']['min']
        rf_max = self.cfg.yaml_data['bands']['rf_hz']['max']
        
        idx_min = int(rf_min / self.grid_step)
        idx_max = int(rf_max / self.grid_step)
        self.mask_lut[idx_min:idx_max] = float(ib_cfg['default_dbc'])
        
        # RF Filter LUT
        self.rf_lut = build_dense_lut(
            self.cfg.rf_filter_raw_freqs,
            self.cfg.rf_filter_raw_atten,
            self.grid_max, self.grid_step, 80.0
        )
        
        self.if2_lut_buffer = np.zeros_like(self.rf_lut)
        self.lo_cache = {} 

    def _get_lo_spectrum(self, lo_def, freq):
        key = (freq, lo_def.name)
        if key not in self.lo_cache:
            self.lo_cache[key] = self.hw.generate_lo_spectrum(lo_def, freq)
        return self.lo_cache[key]

    def evaluate_policy(self, if2_filter: FilterModel, search_mode=True):
        """ 
        Evaluate the entire policy for a fixed IF2 filter. 
        Returns: (worst_margin, retune_count)
        """
        worst_margin = 999.0
        
        # 1. Build IF2 LUT
        self.if2_lut_buffer[:] = float(if2_filter.stop_floor) * -1.0 
        
        if if2_filter.model_type == 0:
             fill_symmetric_filter_lut(
                 self.if2_lut_buffer, 
                 if2_filter.center_hz, if2_filter.bw_hz,
                 if2_filter.passband_il, if2_filter.rolloff,
                 if2_filter.stop_floor, self.grid_step
             )
        elif if2_filter.model_type == 1:
             if len(self.cfg.if2_proto_norm_x) == 0:
                 fill_symmetric_filter_lut(self.if2_lut_buffer, if2_filter.center_hz, if2_filter.bw_hz, 
                                           1.0, 40.0, if2_filter.stop_floor, self.grid_step)
             else:
                 fill_scaled_s2p_lut(
                     self.if2_lut_buffer,
                     if2_filter.center_hz, if2_filter.bw_hz,
                     self.cfg.if2_proto_norm_x, self.cfg.if2_proto_val_y,
                     if2_filter.stop_floor, self.grid_step
                 )

        # Retune/Lock Optimization Logic
        # We track the previously used LO frequencies to detect switches.
        # We also implement a simple hysteresis: prefer previous LO if margin is acceptable.
        prev_lo1 = -1.0
        prev_lo2 = -1.0
        total_switches = 0
        
        # Target margin for hysteresis (if margin > this, don't switch just to get 0.1dB better)
        hysteresis_target = max(0.0, self.opt_cutoff + 5.0) 

        for tile in self.cfg.tiles:
            # Calculate Candidates: [Margin, LO1_Freq, LO2_Freq]
            candidates = []
            
            # 1. Non-Inverting (Sum-Sum) - Low Side / Low Side
            # Note: Depending on frequencies, one mixer might need diff to be low side, 
            # but architecture usually enforces Sum-Sum or Diff-Diff.
            # Here we test the two allowed architectural modes defined in Spec 3.2
            
            # Mode A: Sum-Sum (Usually Low Side Injection if LO < RF)
            m_sum = self._eval_chain_fast(tile, if2_filter, high_side=False, search_mode=search_mode)
            if m_sum > -900:
                candidates.append((m_sum, if2_filter.center_hz - tile.if1_center_hz, tile.rf_center_hz - if2_filter.center_hz))
                
            # Mode B: Diff-Diff (High Side Injection)
            m_diff = self._eval_chain_fast(tile, if2_filter, high_side=True, search_mode=search_mode)
            if m_diff > -900:
                candidates.append((m_diff, if2_filter.center_hz + tile.if1_center_hz, tile.rf_center_hz + if2_filter.center_hz))
            
            if not candidates:
                return -999.0, 9999 # Invalid design
            
            # Sort by Margin descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_cand = candidates[0]
            
            # Hysteresis Logic:
            # Check if we can reuse previous LOs with acceptable margin
            if prev_lo1 > 0:
                found_sticky = False
                for cand in candidates:
                    margin, l1, l2 = cand
                    # Check if LOs match previous (approx equality for floats)
                    if abs(l1 - prev_lo1) < 100 and abs(l2 - prev_lo2) < 100:
                        # If this "sticky" candidate is good enough, take it
                        if margin >= hysteresis_target:
                            best_cand = cand
                            found_sticky = True
                        break
                
            # Update Stats
            chosen_margin, chosen_l1, chosen_l2 = best_cand
            
            if prev_lo1 > 0:
                if abs(chosen_l1 - prev_lo1) > 1000: total_switches += 1
                if abs(chosen_l2 - prev_lo2) > 1000: total_switches += 1
            
            prev_lo1, prev_lo2 = chosen_l1, chosen_l2
            
            if chosen_margin < worst_margin:
                worst_margin = chosen_margin
                
            if search_mode and worst_margin < self.opt_cutoff:
                return worst_margin, 9999

        return worst_margin, total_switches

    def get_all_valid_candidates(self, tile: Tile, if2_filter: FilterModel, min_margin_db=0.0):
        """ Used for final policy generation (Full Physics). Returns explicit side info. """
        results = []
        
        # Re-fill IF2 LUT (Safety)
        self.if2_lut_buffer[:] = float(if2_filter.stop_floor) * -1.0
        if if2_filter.model_type == 0:
             fill_symmetric_filter_lut(self.if2_lut_buffer, if2_filter.center_hz, if2_filter.bw_hz,
                                       if2_filter.passband_il, if2_filter.rolloff, if2_filter.stop_floor, self.grid_step)
        else:
             fill_scaled_s2p_lut(self.if2_lut_buffer, if2_filter.center_hz, if2_filter.bw_hz,
                                 self.cfg.if2_proto_norm_x, self.cfg.if2_proto_val_y,
                                 if2_filter.stop_floor, self.grid_step)

        # Explicitly pass side/mode
        res_sum = self._eval_chain_detailed(tile, if2_filter, high_side=False)
        if res_sum and res_sum[0] >= min_margin_db - 20.0:
            results.append(res_sum)
            
        res_diff = self._eval_chain_detailed(tile, if2_filter, high_side=True)
        if res_diff and res_diff[0] >= min_margin_db - 20.0:
            results.append(res_diff)
            
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def _eval_chain_fast(self, tile, if2_filter, high_side, search_mode=True):
        """ Returns margin (float) or -999 if invalid. """
        if high_side: # Diff: LO > Signal
            lo1_freq = if2_filter.center_hz + tile.if1_center_hz
            lo2_freq = tile.rf_center_hz + if2_filter.center_hz
        else: # Sum: Signal + LO
            lo1_freq = if2_filter.center_hz - tile.if1_center_hz
            lo2_freq = tile.rf_center_hz - if2_filter.center_hz
            
        # 1. LO Validity Checks
        valid1, _, _ = self.hw.get_valid_lo_config(self.hw.lo1_def, lo1_freq, self.hw.mixer1.drive_req)
        if not valid1: return -999.0
        
        valid2, _, _ = self.hw.get_valid_lo_config(self.hw.lo2_def, lo2_freq, self.hw.mixer2.drive_req)
        if not valid2: return -999.0
        
        # 2. Prepare Spectra
        lo1_spec = self._get_lo_spectrum(self.hw.lo1_def, lo1_freq)
        lo2_spec = self._get_lo_spectrum(self.hw.lo2_def, lo2_freq)
        
        # 3. Calculate Stage 1 Intermediates
        stage1_spurs = compute_stage1_intermediates(
            lo1_spec, lo1_freq, tile.if1_center_hz,
            self.hw.mixer1.spur_table_np, not high_side, # If high_side=True (Diff), desired is Sum/Diff? 
            # Logic Check: high_side=True means LO > IF. F_out = LO-IF. This is Diff. 
            # 'if2_side_high' param in kernel controls sum/diff math.
            # compute_stage1_intermediates expects 'if2_side_high'. 
            # If high_side=True (Diff), we pass False? No.
            # Let's check kernel: if if2_side_high: f = lo+if. Else f = |lo-if|.
            # So if we want Diff (high_side), we pass False. If we want Sum, we pass True.
            # previously: `not high_side`.
            # If high_side=False (Sum mode), `not high_side` is True. Kernel calculates lo+if. Correct.
            # If high_side=True (Diff mode), `not high_side` is False. Kernel calculates |lo-if|. Correct.
            self.if2_lut_buffer, self.grid_step, self.noise_floor,
            search_mode
        )
        
        # 4. Calculate Stage 2
        f_if2_desired = if2_filter.center_hz
        
        margin = compute_stage2_from_intermediates(
            stage1_spurs,
            lo2_spec, lo2_freq, tile.rf_center_hz,
            self.hw.mixer2.spur_table_np, not high_side,
            f_if2_desired,
            self.rf_lut, self.mask_lut,
            self.grid_step, self.guard_db, self.noise_floor,
            search_mode
        )
        
        return margin

    def _eval_chain_detailed(self, tile, if2_filter, high_side):
        """ Returns full object tuple for policy storage. Always runs full physics. """
        margin = self._eval_chain_fast(tile, if2_filter, high_side, search_mode=False)
        if margin <= -900.0: return None
        
        if high_side:
            lo1_freq = if2_filter.center_hz + tile.if1_center_hz
            lo2_freq = tile.rf_center_hz + if2_filter.center_hz
        else:
            lo1_freq = if2_filter.center_hz - tile.if1_center_hz
            lo2_freq = tile.rf_center_hz - if2_filter.center_hz
            
        valid1, pad1, p1 = self.hw.get_valid_lo_config(self.hw.lo1_def, lo1_freq, self.hw.mixer1.drive_req)
        valid2, pad2, p2 = self.hw.get_valid_lo_config(self.hw.lo2_def, lo2_freq, self.hw.mixer2.drive_req)
        
        # Store side explicitly for diagnostics
        side_str = "high" if high_side else "low"
        lo1_obj = LO_Candidate(lo1_freq, "frac", side_str, p1, pad1, np.array([]))
        lo2_obj = LO_Candidate(lo2_freq, "frac", side_str, p2, pad2, np.array([]))
        
        return (margin, lo1_obj, lo2_obj)