import numpy as np
from .buc_structures import Tile, FilterModel, LO_Candidate, TilePolicyEntry
from .buc_kernels import (
    build_dense_lut, 
    fill_symmetric_filter_lut, 
    fill_scaled_s2p_lut,
    precompute_mixing_recipes,
    get_lut_val,
    compute_stage1_spurs_no_if2,
    compute_stage2_from_intermediates
)

class SpurEngine:
    BRITTLENESS_STEP_HZ = 1e6

    def __init__(self, config, hw_stack):
        self.cfg = config
        self.hw = hw_stack
        self.grid_max = self.cfg.grid_max_freq_hz
        self.grid_step = self.cfg.grid_step_hz
        self.opt_cutoff = self.cfg.opt_cutoff_db
        self.noise_floor = self.cfg.noise_floor_dbc
        self.guard_db = float(self.cfg.yaml_data['constraints']['guard_margin_db'])
        
        self.rbw_hz = self.cfg.rbw_hz
        self.max_order_stage1 = int(self.cfg.m1_max_order)
        self.max_order_stage2 = int(self.cfg.m2_max_order)
        self.hysteresis_hz = float(self.cfg.hysteresis_hz)
        self.cross_stage_sum_max = int(self.cfg.cross_stage_sum_max)

        self.enforce_non_inverting = bool(
            self.cfg.yaml_data.get('constraints', {}).get('enforce_non_inverting_chain', True)
        )
        if not self.enforce_non_inverting:
            raise NotImplementedError(
                "enforce_non_inverting_chain=False is not supported."
            )
        
        self.stage1_buffer_size = 65536 
        self.stage1_buffer = np.zeros((self.stage1_buffer_size, 4), dtype=np.float64)
        
        self.stage1_raw_buffer_size = 65536
        self.stage1_raw_buffer = np.zeros((self.stage1_raw_buffer_size, 4), dtype=np.float64)
        self.stage1_raw_cache = {}
        
        self._lo1_cand_cache = {}
        self._lo2_cand_cache = {}

        # Build Mask LUT with Slope Support
        mask_oob_def = self.cfg.yaml_data['masks']['outofband']['default_dbc']
        self.mask_lut = np.full(int(self.grid_max/self.grid_step)+2, mask_oob_def, dtype=np.float32)
        
        ib_cfg = self.cfg.yaml_data['masks']['inband']
        rf_min, rf_max = self.cfg.yaml_data['bands']['rf_hz']['min'], self.cfg.yaml_data['bands']['rf_hz']['max']
        idx_min, idx_max = int(rf_min / self.grid_step), int(rf_max / self.grid_step)
        if idx_min < len(self.mask_lut) and idx_max < len(self.mask_lut):
            self.mask_lut[idx_min:idx_max] = float(ib_cfg['default_dbc'])

            
        def _apply_mask_table(table, default_step, lut):
            for entry in table:
                if 'freq_hz' in entry and 'limit_dbc' in entry:
                    f = float(entry['freq_hz'])
                    idx = int(f / default_step)
                    if 0 <= idx < len(lut):
                        lut[idx] = float(entry['limit_dbc'])
                elif all(k in entry for k in ('start_hz', 'stop_hz', 'limit_dbc')):
                    f0 = float(entry['start_hz'])
                    f1 = float(entry['stop_hz'])
                    idx0 = max(0, int(f0 / default_step))
                    idx1 = min(len(lut), int(f1 / default_step))
                    lut[idx0:idx1] = float(entry['limit_dbc'])
                # Support Sloped Masks
                elif all(k in entry for k in ('start_hz', 'stop_hz', 'start_limit_dbc', 'stop_limit_dbc')):
                    f0 = float(entry['start_hz'])
                    f1 = float(entry['stop_hz'])
                    l0 = float(entry['start_limit_dbc'])
                    l1 = float(entry['stop_limit_dbc'])
                    idx0 = max(0, int(f0 / default_step))
                    idx1 = min(len(lut), int(f1 / default_step))
                    length = max(1, idx1 - idx0)
                    for i in range(idx0, idx1):
                        t = (i - idx0) / max(1, length - 1)
                        lut[i] = l0 + t * (l1 - l0)

        _apply_mask_table(ib_cfg.get('table', []), self.grid_step, self.mask_lut)
        _apply_mask_table(self.cfg.yaml_data['masks']['outofband'].get('table', []),
                          self.grid_step, self.mask_lut)
        
        self.rf_lut = build_dense_lut(
            self.cfg.rf_filter_raw_freqs,
            self.cfg.rf_filter_raw_atten,
            self.grid_max, self.grid_step, 80.0
        )
        self.rf_passband_lo_hz, self.rf_passband_hi_hz = self._estimate_rf_passband(
            self.rf_lut, self.grid_step
        )
        
        self.if2_lut_buffer = np.zeros_like(self.rf_lut)
        self.lo_cache = {}      
        self.recipe_cache = {} 
        self._warmup_jit()

    def _eval_if_iso_mixer2_leakage(self, if2_filter: FilterModel) -> float:
        """
        Model Mixer-2 IF→RF isolation as a direct RF-port spur that
        appears near the IF2 centre frequency, then is shaped by the
        RF BPF and checked against the mask.

        Returns a "margin" for this path. The caller should take the
        minimum of this and the main spur margin.
        """
        # Respect the global isolation flag
        if not self.hw.mixer2.include_isolation_spurs:
            return 999.0  # effectively no constraint

        f_if2 = float(if2_filter.center_hz)
        if f_if2 <= 0.0:
            return 999.0

        # Check against LUT extent
        idx = int(f_if2 / self.grid_step)
        if idx < 0 or idx >= self.mask_lut.shape[0]:
            return 999.0

        # IF→RF isolation is already defined as a RF-port spur level
        iso_db = self.hw.mixer2.if_feedthrough_rej_db()  # e.g. -45 dBc

        # RF filter attenuation at that IF2-like frequency
        atten_rf = get_lut_val(f_if2, self.rf_lut, self.grid_step)
        final_lvl = iso_db - atten_rf  # dBc after RF BPF

        # If well below noise floor, it is irrelevant
        if final_lvl < self.noise_floor:
            return 999.0

        limit = self.mask_lut[idx]
        margin = limit - final_lvl - self.guard_db
        return float(margin)


    def _estimate_rf_passband(self, lut, step_hz):
        """Config-driven estimation with LUT validation."""
        rf_cfg = self.cfg.yaml_data['bands']['rf_hz']
        rf_min_cfg, rf_max_cfg = rf_cfg['min'], rf_cfg['max']
        idx_min_cfg, idx_max_cfg = int(rf_min_cfg / step_hz), int(rf_max_cfg / step_hz)

        max_inband = self.cfg.rf_passband_max_atten_db

        lo_idx, hi_idx = -1, -1
        limit = min(idx_max_cfg, len(lut))
        
        for i in range(idx_min_cfg, limit):
            if lut[i] <= max_inband:
                if lo_idx < 0: lo_idx = i
                hi_idx = i

        if lo_idx < 0:
            return rf_min_cfg, rf_max_cfg

        f_lo = lo_idx * step_hz
        f_hi = hi_idx * step_hz
        
        cfg_span = rf_max_cfg - rf_min_cfg
        lut_span = f_hi - f_lo

        if lut_span < 0.2 * cfg_span:
            print("Warning: RF LUT passband narrower than 20% of config span; using config RF range.")
            return rf_min_cfg, rf_max_cfg

        return f_lo, f_hi

    def _warmup_jit(self):
        dummy_lut = np.zeros(100, dtype=np.float32)
        get_lut_val(50.0, dummy_lut, 1.0)
        dummy_recipes = np.zeros((10, 6), dtype=np.float64)
        dummy_stage1 = np.zeros((10, 4), dtype=np.float64)
        compute_stage1_spurs_no_if2(dummy_recipes, 1e9, dummy_stage1)
        compute_stage2_from_intermediates(
            dummy_stage1, dummy_recipes, 30e9, 5e9, 
            dummy_lut, dummy_lut, 1e6, 2.0, -100.0, 1e6, 12, 0.0
        )

    def _get_lo_data(self, lo_def, freq, mixer_model, delivered_power, search_mode, max_order, dominant_only=False):
        p_round = round(delivered_power, 2)
        # Cache Key includes dominant_only flag
        key = (
            freq, lo_def.name, p_round, int(search_mode), max_order, 
            lo_def.mode_name, 1, 1, int(dominant_only)
        )
        
        if key not in self.lo_cache:
            # Generate spec using f_pfd from model (Phase-1 assumption)
            spec = self.hw.generate_lo_spectrum(lo_def, freq)
            drive_delta = delivered_power - mixer_model.nom_drive_dbm
            recipes = precompute_mixing_recipes(
                spec, 
                mixer_model.spur_table_np, 
                search_mode,
                drive_delta,
                mixer_model.scaling_slope,
                mixer_model.scaling_cap,
                max_order,
                include_lo_feedthrough=mixer_model.include_isolation_spurs,
                lo_feedthrough_rej_db=mixer_model.lo_feedthrough_rej_db(),
                dominant_only=dominant_only
            )
            self.lo_cache[key] = spec
            self.recipe_cache[key] = recipes
            
        return self.lo_cache[key], self.recipe_cache[key]

    def _quantize_lo_freq(self, lo_model, target_freq):
        step = lo_model.step_hz
        if step <= 0.0: return target_freq
        n = round(target_freq / step)
        return n * step

    def _get_lo_candidate(self, lo_model, cache, target_freq, mixer_model):
        q_freq = self._quantize_lo_freq(lo_model, target_freq)
        key = q_freq
        if key in cache: return cache[key]

        valid, pad, p_del = self.hw.get_valid_lo_config(
            lo_model, q_freq, mixer_model.drive_req
        )
        if not valid:
            cache[key] = None
            return None

        cand = LO_Candidate(
            freq_hz=q_freq,
            mode=lo_model.mode_name,
            side="unknown",
            delivered_power_dbm=p_del,
            pad_db=pad,
            spectrum=np.empty((0, 2), dtype=np.float64),
        )
        cache[key] = cand
        return cand

    def _get_stage1_spurs_raw(self, tile, lo1_freq, is_sum_mix, recipes1, search_mode, dominant_only: bool):
        # Cache key explicitly includes IF1 center and dominant_only flag.
        # This ensures that if the same Tile ID is reused with different IF1
        # frequencies or with dominant-only vs full recipes, we do not return
        # stale spurs.
        key = (
            tile.id, 
            lo1_freq, 
            tile.if1_center_hz, 
            int(search_mode), 
            1 if is_sum_mix else 0,
            int(dominant_only),
        )
        if key in self.stage1_raw_cache:
            return self.stage1_raw_cache[key]

        while True:
            count = compute_stage1_spurs_no_if2(
                recipes1,
                tile.if1_center_hz,
                self.stage1_raw_buffer,
            )
            if count < 0:
                new_size = self.stage1_raw_buffer_size * 2
                self.stage1_raw_buffer = np.zeros((new_size, 4), dtype=np.float64)
                self.stage1_raw_buffer_size = new_size
                continue
            break

        arr = self.stage1_raw_buffer[:count].copy()
        self.stage1_raw_cache[key] = arr
        return arr

    # --- Unified Policy Builder ---
    def build_policy_for_if2(
        self,
        if2_filter: FilterModel,
        search_mode: bool = True,
        compute_brittleness: bool = False,
        stop_if_margin_below: float = None,
    ):
        """
        Construct policy with hysteresis, heuristic lock time, and pruning.
        """
        # 1. Prepare IF2 LUT
        self.if2_lut_buffer[:] = float(if2_filter.stop_floor)
        if if2_filter.model_type == 0:
            fill_symmetric_filter_lut(
                self.if2_lut_buffer, if2_filter.center_hz, if2_filter.bw_hz,
                if2_filter.passband_il, if2_filter.rolloff, if2_filter.stop_floor,
                self.grid_step
            )
        elif if2_filter.model_type == 1 and len(self.cfg.if2_proto_norm_x) > 0:
            fill_scaled_s2p_lut(
                self.if2_lut_buffer, if2_filter.center_hz, if2_filter.bw_hz,
                self.cfg.if2_proto_norm_x, self.cfg.if2_proto_val_y,
                if2_filter.stop_floor, self.grid_step
            )

        # 2. Determine Traversal Order (RF-snaked)
        sorted_tiles = sorted(self.cfg.tiles, key=lambda t: (t.rf_center_hz, t.if1_center_hz))
        grouped = {}
        for t in sorted_tiles:
            grouped.setdefault(t.rf_center_hz, []).append(t)
        rf_keys = sorted(grouped.keys())
        processing_order = []
        for i, rf in enumerate(rf_keys):
            group = grouped[rf]
            if i % 2 == 1: group = group[::-1]
            processing_order.extend(group)

        worst_margin = 999.0
        total_lock_time = 0.0
        retune_count = 0
        
        prev_lo1_freq = None
        prev_lo2_freq = None
        prev_side = None
        
        entries = []

        for tile in processing_order:
            # --- Dominant Pruning ---
            # Check Low Side (Sum-Sum)
            margin_low_approx = self._eval_chain_approx(tile, if2_filter, high_side=False)
            low_viable = True
            if margin_low_approx < (self.cfg.dominant_prune_cutoff_db - self.cfg.dominant_spur_margin_buffer_db):
                low_viable = False
                
            # Check High Side (Diff-Diff)
            margin_high_approx = self._eval_chain_approx(tile, if2_filter, high_side=True)
            high_viable = True
            if margin_high_approx < (self.cfg.dominant_prune_cutoff_db - self.cfg.dominant_spur_margin_buffer_db):
                high_viable = False

            candidates = []

            # --- Full (or Search-Mode) Eval for Viable Sides ---
            if low_viable:
                m_sum = self._eval_chain_fast(tile, if2_filter, high_side=False, search_mode=search_mode)
                if m_sum > -900.0:
                    lo1_t = if2_filter.center_hz - tile.if1_center_hz
                    lo2_t = tile.rf_center_hz - if2_filter.center_hz
                    c1 = self._get_lo_candidate(self.hw.lo1_def, self._lo1_cand_cache, lo1_t, self.hw.mixer1)
                    c2 = self._get_lo_candidate(self.hw.lo2_def, self._lo2_cand_cache, lo2_t, self.hw.mixer2)
                    if c1 and c2:
                        candidates.append(("low", m_sum, c1, c2))
            
            if high_viable:
                m_diff = self._eval_chain_fast(tile, if2_filter, high_side=True, search_mode=search_mode)
                if m_diff > -900.0:
                    lo1_t = if2_filter.center_hz + tile.if1_center_hz
                    lo2_t = tile.rf_center_hz + if2_filter.center_hz
                    c1 = self._get_lo_candidate(self.hw.lo1_def, self._lo1_cand_cache, lo1_t, self.hw.mixer1)
                    c2 = self._get_lo_candidate(self.hw.lo2_def, self._lo2_cand_cache, lo2_t, self.hw.mixer2)
                    if c1 and c2:
                        candidates.append(("high", m_diff, c1, c2))

            if not candidates:
                return -999.0, 99999.0, 0, []

            candidates.sort(key=lambda x: x[1], reverse=True)
            best_side, best_margin, best_lo1, best_lo2 = candidates[0]

            # --- Hysteresis Logic ---
            if prev_lo1_freq is not None:
                for side, margin, l1, l2 in candidates:
                    d1 = abs(l1.freq_hz - prev_lo1_freq)
                    d2 = abs(l2.freq_hz - prev_lo2_freq)
                    # If within hysteresis and margin acceptable, prefer sticking
                    if d1 < self.hysteresis_hz and d2 < self.hysteresis_hz:
                        if margin >= self.opt_cutoff + 5.0:
                            best_side, best_margin, best_lo1, best_lo2 = side, margin, l1, l2
                        break
            
            # --- Lock Time Heuristic ---
            lock_ms = 0.0
            did_retune = False
            if prev_lo1_freq is not None:
                d1_mhz = abs(best_lo1.freq_hz - prev_lo1_freq) / 1e6
                d2_mhz = abs(best_lo2.freq_hz - prev_lo2_freq) / 1e6
                t1 = self.hw.lo1_def.lock_base_ms + d1_mhz * self.hw.lo1_def.lock_per_mhz_ms
                t2 = self.hw.lo2_def.lock_base_ms + d2_mhz * self.hw.lo2_def.lock_per_mhz_ms
                lock_ms = max(t1, t2)
                if d1_mhz > 0.0 or d2_mhz > 0.0:
                    did_retune = True
            
            prev_lo1_freq = best_lo1.freq_hz
            prev_lo2_freq = best_lo2.freq_hz
            prev_side = best_side
            
            total_lock_time += lock_ms
            if did_retune: retune_count += 1
            
            if best_margin < worst_margin:
                worst_margin = best_margin

            # Early Abort
            if stop_if_margin_below is not None and worst_margin < stop_if_margin_below:
                return worst_margin, 99999.0, retune_count, []

            # Record Entry
            brittleness = 0.0
            if compute_brittleness:
                brittleness = self.calculate_brittleness(tile, if2_filter, (best_side == "high"))

            c1_final = LO_Candidate(
                best_lo1.freq_hz, best_lo1.mode, best_side, 
                best_lo1.delivered_power_dbm, best_lo1.pad_db, best_lo1.spectrum
            )
            c2_final = LO_Candidate(
                best_lo2.freq_hz, best_lo2.mode, best_side,
                best_lo2.delivered_power_dbm, best_lo2.pad_db, best_lo2.spectrum
            )

            entry = TilePolicyEntry(
                tile_id=tile.id,
                if1_center_hz=tile.if1_center_hz,
                bw_hz=tile.bw_hz,
                rf_center_hz=tile.rf_center_hz,
                lo1=c1_final,
                lo2=c2_final,
                spur_margin_db=best_margin,
                brittleness_db_per_step=brittleness,
                lock_time_ms_tile=lock_ms,
                retune_occurred=did_retune,
                side=best_side
            )
            entries.append(entry)

        entries.sort(key=lambda e: e.tile_id)
        return worst_margin, total_lock_time, retune_count, entries

    def evaluate_policy(self, if2_filter: FilterModel, search_mode: bool = True):
        """
        Entry point used during IF2 search and diagnostics.
        Returns (score, expected_lock_ms).
        """
        cutoff = self.cfg.opt_cutoff_db if search_mode else None

        worst_margin, total_lock, retunes, entries = self.build_policy_for_if2(
            if2_filter,
            search_mode=search_mode,
            compute_brittleness=False,
            stop_if_margin_below=cutoff,
        )
        self.last_retune_count = retunes

        if not entries and worst_margin <= -900:
            return -999.0, 99999.0

        # Pass retunes to scoring function now
        score, expected_lock = self._score_policy(if2_filter, worst_margin, entries, retunes)
        return score, expected_lock

    def _aggregate_expected_lock(self, entries: list[TilePolicyEntry]) -> float:
        if not entries: return 0.0
        n = len(entries)
        lock_by_id = np.zeros(len(self.cfg.tiles), dtype=float)
        for e in entries:
            lock_by_id[e.tile_id] = e.lock_time_ms_tile

        P = self.cfg.markov_matrix
        pi = self.cfg.markov_stationary

        # Markov-weighted heuristic
        if P.size > 0 and pi is not None and len(pi) == len(lock_by_id):
            return float(np.dot(pi, lock_by_id))
        else:
            # Fallback: uniform average
            return float(lock_by_id.mean())

    def _score_policy(self, if2_filter, worst_margin_db, entries, retune_count=0):
        # Hard minimum constraint
        if worst_margin_db < self.cfg.min_margin_db:
            return -1e9, 1e9

        expected_lock = self._aggregate_expected_lock(entries)
        
        lam_lock = self.cfg.runtime_lock_time_weight
        lam_retune = self.cfg.retune_penalty_weight if self.cfg.prefer_fewer_retunes else 0.0
        
        score = worst_margin_db - (lam_lock * expected_lock) - (lam_retune * float(retune_count))
        return score, expected_lock

    def calculate_brittleness(self, tile: Tile, if2_filter: FilterModel, high_side: bool):
        base = self._eval_chain_fast(tile, if2_filter, high_side, search_mode=False)
        if base <= -900: return 99.9
        step = self.BRITTLENESS_STEP_HZ 
        t_if1 = Tile(tile.id, tile.if1_center_hz + step, tile.bw_hz, tile.rf_center_hz)
        m_if1 = self._eval_chain_fast(t_if1, if2_filter, high_side, search_mode=False)
        t_rf = Tile(tile.id, tile.if1_center_hz, tile.bw_hz, tile.rf_center_hz + step)
        m_rf = self._eval_chain_fast(t_rf, if2_filter, high_side, search_mode=False)
        return max(max(0, base - m_if1), max(0, base - m_rf))
    
    def _eval_chain_approx(self, tile, if2_filter, high_side: bool):
        """Approximate margin using dominant-only spur set."""
        return self._eval_chain_fast(
            tile, if2_filter, high_side, search_mode=True, dominant_only=True
        )

    def _eval_chain_fast(self, tile, if2_filter, high_side, search_mode=True, dominant_only=False):
        is_sum_mix = not high_side
        
        if high_side:
            lo1_target = if2_filter.center_hz + tile.if1_center_hz
            lo2_target = tile.rf_center_hz + if2_filter.center_hz
        else:
            lo1_target = if2_filter.center_hz - tile.if1_center_hz
            lo2_target = tile.rf_center_hz - if2_filter.center_hz

        lo1_freq = self._quantize_lo_freq(self.hw.lo1_def, lo1_target)
        lo2_freq = self._quantize_lo_freq(self.hw.lo2_def, lo2_target)
        
        r1 = self.hw.lo1_def.freq_range
        if not (r1[0] <= lo1_freq <= r1[1]): return -999.0
        r2 = self.hw.lo2_def.freq_range
        if not (r2[0] <= lo2_freq <= r2[1]): return -999.0
            
        valid1, _, p1 = self.hw.get_valid_lo_config(self.hw.lo1_def, lo1_freq, self.hw.mixer1.drive_req)
        if not valid1: return -999.0
        valid2, _, p2 = self.hw.get_valid_lo_config(self.hw.lo2_def, lo2_freq, self.hw.mixer2.drive_req)
        if not valid2: return -999.0

        if self._early_reject_chain(tile, if2_filter, lo1_freq, lo2_freq, high_side):
            return -999.0
        
        _, recipes1 = self._get_lo_data(
            self.hw.lo1_def, lo1_freq, self.hw.mixer1, p1, search_mode, 
            self.max_order_stage1, dominant_only
        )
        _, recipes2 = self._get_lo_data(
            self.hw.lo2_def, lo2_freq, self.hw.mixer2, p2, search_mode, 
            self.max_order_stage2, dominant_only
        )
        
        # IF2-Agnostic Stage 1 Spurs
        raw_stage1 = self._get_stage1_spurs_raw(
            tile, lo1_freq, is_sum_mix, recipes1, search_mode, dominant_only
        )
        
        if raw_stage1.shape[0] > self.stage1_buffer_size:
             new_size = max(self.stage1_buffer_size * 2, raw_stage1.shape[0])
             self.stage1_buffer = np.zeros((new_size, 4), dtype=np.float64)
             self.stage1_buffer_size = new_size
             
        # Apply IF2 Filter & Noise Floor
        # AND explicit desired path suppression based on geometry
        count = 0
        for i in range(raw_stage1.shape[0]):
             f_if2 = raw_stage1[i, 0]
             lvl_pre = raw_stage1[i, 1]
             m1_abs = raw_stage1[i, 2]
             n1_abs = raw_stage1[i, 3]
             
             atten_if2 = get_lut_val(f_if2, self.if2_lut_buffer, self.grid_step)
             lvl_post = lvl_pre - atten_if2
             
             if lvl_post < self.noise_floor: continue
             
             self.stage1_buffer[count, 0] = f_if2
             self.stage1_buffer[count, 1] = lvl_post
             self.stage1_buffer[count, 2] = m1_abs
             self.stage1_buffer[count, 3] = n1_abs
             count += 1
             
        # IF Feedthrough Injection
        #if self.hw.mixer2.include_isolation_spurs:
        #     if2_rej = self.hw.mixer2.if_feedthrough_rej_db()
        #     if2_lvl = 0.0 + if2_rej # desired IF2 reference
        #     if count < self.stage1_buffer_size:
        #         self.stage1_buffer[count, 0] = if2_filter.center_hz
        #         self.stage1_buffer[count, 1] = if2_lvl
        #         self.stage1_buffer[count, 2] = 0.0 # m1
        #         self.stage1_buffer[count, 3] = 1.0 # n1
        #         count += 1

        # EXPLICIT DESIRED (1,1) SUPPRESSION
        if self.cfg.enforce_desired_mn11_only:
            if high_side:
                # Diff-Diff desired: |LO1 - IF1|
                f_if2_desired = abs(lo1_freq - tile.if1_center_hz)
            else:
                # Sum-Sum desired: LO1 + IF1
                f_if2_desired = lo1_freq + tile.if1_center_hz

            # Tolerance check
            tol_if2 = max(100.0, self.grid_step * 1.0)
            
            filtered_count = 0
            for i in range(count):
                f_if2 = self.stage1_buffer[i, 0]
                m1_abs = self.stage1_buffer[i, 2]
                n1_abs = self.stage1_buffer[i, 3]
                
                # Drop only the (1,1) tone at the desired IF2 frequency
                if (m1_abs == 1.0 and n1_abs == 1.0 and abs(f_if2 - f_if2_desired) < tol_if2):
                    continue
                    
                self.stage1_buffer[filtered_count, :] = self.stage1_buffer[i, :]
                filtered_count += 1
            count = filtered_count

        valid_stage1 = self.stage1_buffer[:count]
        
        margin = compute_stage2_from_intermediates(
            valid_stage1,
            recipes2,
            tile.rf_center_hz,
            if2_filter.center_hz,
            self.rf_lut, self.mask_lut,
            self.grid_step, self.guard_db, self.noise_floor,
            self.rbw_hz,
            self.cross_stage_sum_max,
            self.cfg.max_spur_level_dbc
        )

        # If the main path already failed hard, just return it
        if margin <= -900.0:
            return margin

        # Include Mixer-2 IF→RF isolation as a direct RF-port spur
        iso_margin = self._eval_if_iso_mixer2_leakage(if2_filter)
        if iso_margin < margin:
            margin = iso_margin

        return margin

    def _early_reject_chain(self, tile, if2_filter, lo1_freq, lo2_freq, high_side: bool) -> bool:
        er_cfg = self.cfg.yaml_data.get('early_reject', {})
        reject_image_if2 = bool(er_cfg.get('image_in_if2_passband', False))
        reject_loft = bool(er_cfg.get('loft_in_if2_or_rf_passbands', False))
        reject_rf_image = bool(er_cfg.get('rf_first_order_image_in_passband', False))

        if not (reject_image_if2 or reject_loft or reject_rf_image):
            return False

        if_bw_half = if2_filter.bw_hz * 0.5
        if2_lo = if2_filter.center_hz - if_bw_half
        if2_hi = if2_filter.center_hz + if_bw_half

        if self.rf_passband_lo_hz > 0.0 or self.rf_passband_hi_hz > 0.0:
            rf_lo = self.rf_passband_lo_hz
            rf_hi = self.rf_passband_hi_hz
        else:
            rf_bw_half = tile.bw_hz * 0.5
            rf_lo = tile.rf_center_hz - rf_bw_half
            rf_hi = tile.rf_center_hz + rf_bw_half

        if reject_image_if2:
            # Sense-aware IF2 image rejection
            if high_side:
                # Diff-Diff: desired = |LO1 - IF1|, image = LO1 + IF1
                f_if2_image = lo1_freq + tile.if1_center_hz
            else:
                # Sum-Sum: desired = LO1 + IF1, image = |LO1 - IF1|
                f_if2_image = abs(lo1_freq - tile.if1_center_hz)

            if if2_lo <= f_if2_image <= if2_hi: return True

        if reject_loft:
            if if2_lo <= lo1_freq <= if2_hi: return True
            if rf_lo <= lo2_freq <= rf_hi: return True

        if reject_rf_image:
            if high_side: f_img = lo2_freq + if2_filter.center_hz
            else: f_img = abs(lo2_freq - if2_filter.center_hz)
            if rf_lo <= f_img <= rf_hi: return True

        return False