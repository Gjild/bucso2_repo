import yaml
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class Tile:
    id: int
    if1_center_hz: float
    bw_hz: float
    rf_center_hz: float

@dataclass
class LO_Candidate:
    freq_hz: float
    mode: str 
    side: str 
    delivered_power_dbm: float
    pad_db: float
    spectrum: np.ndarray 

@dataclass
class TilePolicyEntry:
    tile_id: int
    if1_center_hz: float
    bw_hz: float
    rf_center_hz: float
    
    lo1: LO_Candidate
    lo2: LO_Candidate
    
    spur_margin_db: float
    brittleness_db_per_step: float
    
    lock_time_ms_tile: float
    retune_occurred: bool
    side: str  # "low" or "high"

@dataclass
class MixerModel:
    name: str
    lo_range: Tuple[float, float]
    drive_req: Tuple[float, float]
    isolation: Dict[str, float]
    spur_table_raw: List[Tuple[int, int, float]]
    # scaling parameters (TDS 5.2)
    nom_drive_dbm: float = 13.0
    scaling_slope: float = 1.0
    scaling_cap: float = 12.0
    
    include_isolation_spurs: bool = True
    
    spur_table_np: np.ndarray = field(init=False)

    def __post_init__(self):
        arr = np.array(self.spur_table_raw, dtype=np.float64)
        if arr.size == 0:
            self.spur_table_np = arr
            return

        # PHASE 1 UPDATE: Do NOT strip (1,1) entries. 
        # The kernel will identify desired vs spur based on config.
        self.spur_table_np = arr
        
    def lo_feedthrough_rej_db(self) -> float:
        return float(self.isolation.get("lo_to_rf_db", -40.0))

    def if_feedthrough_rej_db(self) -> float:
        return float(self.isolation.get("if_to_rf_db", -60.0))

@dataclass
class SynthesizerModel:
    name: str
    freq_range: Tuple[float, float]
    step_hz: float
    power_freqs: np.ndarray
    power_dbm: np.ndarray
    dist_loss_db: float
    pad_options: List[float]
    harmonics: List[Dict] 
    pfd_freq_hz: float
    pfd_spurs: List[Dict]
    frac_boundary_enabled: bool
    frac_boundary_lvl: float
    frac_boundary_slope: float
    # Explicit PFD range (Hz)
    pfd_min_hz: float = 0.0
    pfd_max_hz: float = 0.0
    # Lock Time Params
    lock_base_ms: float = 0.4
    lock_per_mhz_ms: float = 0.002
    
    # Scaffolding fields
    vco_dividers: List[int] = field(default_factory=lambda: [1])
    pfd_dividers: List[int] = field(default_factory=lambda: [1])
    mode_name: str = "fracN"
    int_frac_switch_penalty_ms: float = 0.0
    
    divider_spectrum: Dict = field(default_factory=dict)

@dataclass
class FilterModel:
    center_hz: float = 0.0
    bw_hz: float = 0.0
    model_type: int = 0  # 0=Symmetric, 1=S2P
    passband_il: float = 1.0        # +ve insertion loss [dB]
    rolloff: float = 40.0           # dB/dec
    stop_floor: float = 80.0        # +ve attenuation [dB]
    proto_idx: int = -1 

@dataclass
class GlobalConfig:
    """
    Global configuration and precomputed grids.
    """
    yaml_data: dict
    tiles: List[Tile] = field(default_factory=list)
    
    grid_max_freq_hz: float = 65e9
    grid_step_hz: float = 500e3 
    
    # Plumbing
    if2_grid_step_hz: float = 500e3
    rf_grid_step_hz: float = 500e3
    mask_grid_step_hz: float = 500e3

    # Targets & Cutoffs
    min_margin_db: float = 0.0
    opt_cutoff_db: float = -20.0
    noise_floor_dbc: float = -100.0
    
    # Physics Knobs
    max_spur_level_dbc: float = 0.0
    rf_passband_max_atten_db: float = 5.0
    
    # RBW & order & hysteresis
    rbw_hz: float = 500e3
    m1_max_order: int = 7
    m2_max_order: int = 7
    hysteresis_hz: float = 50e6
    cross_stage_sum_max: int = 12
    
    # Desired Path Config
    enforce_desired_mn11_only: bool = True
    desired_stage1_mn: Tuple[int, int] = (1, 1)
    desired_stage2_mn: Tuple[int, int] = (1, 1)

    rf_filter_raw_freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    rf_filter_raw_atten: np.ndarray = field(default_factory=lambda: np.array([]))
    
    if2_proto_norm_x: np.ndarray = field(default_factory=lambda: np.array([]))
    if2_proto_val_y: np.ndarray = field(default_factory=lambda: np.array([]))

    # New: IF1 harmonic model as list of (k, rel_dBc) pairs.
    # k : positive integer harmonic index (1 = fundamental).
    # rel_dBc : amplitude of that harmonic relative to the fundamental (0 dBc).
    if1_harmonics: List[Tuple[int, float]] = field(default_factory=list)
    
    # Runtime / Pruning
    runtime_lock_time_weight: float = 0.0
    retune_penalty_weight: float = 0.0
    prefer_fewer_retunes: bool = True
    dominant_prune_cutoff_db: float = -20.0
    dominant_spur_margin_buffer_db: float = 2.0
    
    markov_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    markov_stationary: Optional[np.ndarray] = None

    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        cfg = GlobalConfig(yaml_data=data)
        cfg._generate_tiles()
        
        grid_cfg = data.get('rbw_binning', {})
        grid_glob = data.get('grids', {})
        
        # Grid steps plumbing
        base_step = float(grid_cfg.get('lut_step_hz', 500e3))
        cfg.if2_grid_step_hz = float(grid_cfg.get('if2_lut_step_hz', base_step))
        cfg.rf_grid_step_hz = float(grid_cfg.get('rf_lut_step_hz', cfg.if2_grid_step_hz))
        cfg.mask_grid_step_hz = float(grid_cfg.get('mask_lut_step_hz', cfg.rf_grid_step_hz))
        
        # Phase-1 simplification: enforce a single grid step for all LUTs
        if not (cfg.if2_grid_step_hz == cfg.rf_grid_step_hz == cfg.mask_grid_step_hz):
            print(
                "Warning: distinct grid steps requested for IF2/RF/mask, "
                "but Phase-1 engine uses a common step. "
                f"Using rf_grid_step_hz={cfg.rf_grid_step_hz} for all LUTs."
            )
        cfg.grid_step_hz = cfg.rf_grid_step_hz
        
        cfg.grid_max_freq_hz = float(grid_glob.get('grid_max_freq_hz', 65e9))
        cfg.rbw_hz = float(grid_cfg.get('rbw_hz', cfg.rf_grid_step_hz))

        orders_cfg = data.get('orders', {})
        cfg.m1_max_order = int(orders_cfg.get('m1n1_max_abs', 7))
        cfg.m2_max_order = int(orders_cfg.get('m2n2_max_abs', 7))
        cfg.cross_stage_sum_max = int(orders_cfg.get('cross_stage_sum_max', 12))
        
        run_set = data.get('runtime_settings', {})
        cfg.opt_cutoff_db = float(run_set.get('optimization_cutoff_db', -20.0))
        cfg.noise_floor_dbc = float(run_set.get('noise_floor_cutoff_dbc', -100.0))
        cfg.hysteresis_hz = float(run_set.get('hysteresis_hz', 50e6))
        cfg.max_spur_level_dbc = float(run_set.get('max_spur_level_dbc', 0.0))
        cfg.rf_passband_max_atten_db = float(run_set.get('rf_passband_max_atten_db', 5.0))
        cfg.runtime_lock_time_weight = float(run_set.get('runtime_lock_time_weight', 0.0))
        cfg.retune_penalty_weight = float(run_set.get('retune_penalty_weight', 0.0))
        cfg.dominant_prune_cutoff_db = float(run_set.get('dominant_prune_cutoff_db', -20.0))
        cfg.dominant_spur_margin_buffer_db = float(run_set.get('dominant_spur_margin_buffer_db', 2.0))

        targ = data.get('targets', {})
        cfg.min_margin_db = float(targ.get('min_margin_db', 0.0))

        # ------------------------------------------------------------------
        # IF1 harmonic spectrum config
        # ------------------------------------------------------------------
        if1_model = data.get("if1_model", {}) or {}
        harmonics_cfg = if1_model.get("harmonics", []) or []

        harmonics: List[Tuple[int, float]] = []
        for idx, h in enumerate(harmonics_cfg):
            try:
                k = int(h.get("k", 1))
                rel = float(h.get("rel_dBc", 0.0))
            except Exception as ex:
                print(
                    f"Warning: skipping malformed if1_model.harmonics[{idx}]: {h!r} "
                    f"(error: {ex})"
                )
                continue

            if k < 1:
                print(
                    f"Warning: skipping IF1 harmonic with non-positive k={k!r} "
                    f"in if1_model.harmonics[{idx}]"
                )
                continue

            harmonics.append((k, rel))

        if not harmonics:
            # Backwards-compatible default: ideal single-tone IF1.
            harmonics = [(1, 0.0)]
            print(
                "Info: no valid IF1 harmonics configured; "
                "defaulting to [(1, 0.0)] (ideal single-tone IF1)."
            )

        cfg.if1_harmonics = harmonics
        # ------------------------------------------------------------------

        # Constraints & Desired Path
        con = data.get('constraints', {})
        cfg.enforce_desired_mn11_only = bool(con.get('enforce_desired_mn11_only', True))
        cfg.desired_stage1_mn = tuple(con.get('desired_stage1_mn', [1, 1]))
        cfg.desired_stage2_mn = tuple(con.get('desired_stage2_mn', [1, 1]))
        
        if cfg.desired_stage1_mn != (1, 1) or cfg.desired_stage2_mn != (1, 1):
            raise NotImplementedError(
                "Only (1,1) desired orders are supported in this phase. "
                "Non-(1,1) desired m,n will be added in a later phase."
            )

        # runtime_policy overrides
        rt_policy = data.get('runtime_policy', {})
        if 'hysteresis_hz' in rt_policy:
            cfg.hysteresis_hz = float(rt_policy['hysteresis_hz'])
        cfg.prefer_fewer_retunes = bool(rt_policy.get('prefer_fewer_retunes', True))
        
        mtx_path = rt_policy.get('markov_transition_matrix_csv')
        if mtx_path:
            try:
                cfg.markov_matrix = np.loadtxt(mtx_path, delimiter=',')
                cfg._compute_stationary()
            except Exception as e:
                print(f"Warning: failed to load Markov matrix '{mtx_path}': {e}")

        # Load Hardware Filters
        hw_choices = data.get('hardware_choices', {}).get('stacks', [])
        rf_file = hw_choices[0].get('rf_bpf_file', None) if hw_choices else None
        cfg._build_filter_profile(rf_file, is_rf=True, strict=False) 
        
        # Load IF2 Prototype if enabled
        if2_opts = data.get('if2_model', {}).get('scaled_s2p', {})
        if if2_opts.get('enabled', False):
            proto_file = if2_opts.get('prototype_s2p_file')
            cfg._build_filter_profile(proto_file, is_rf=False, strict=True)
            
        return cfg

    def _compute_stationary(self):
        P = self.markov_matrix
        if P.size == 0:
            self.markov_stationary = None
            return

        n = P.shape[0]
        if n != len(self.tiles):
            print(f"Warning: Markov matrix size {n} != num_tiles {len(self.tiles)}")
            return

        dist = np.ones(n) / n
        for _ in range(1000):
            dist_new = dist @ P
            if np.max(np.abs(dist_new - dist)) < 1e-10:
                dist = dist_new
                break
            dist = dist_new
        self.markov_stationary = dist

    def _generate_tiles(self):
        g = self.yaml_data['grids']
        b = self.yaml_data['bands']
        if1_min, if1_max = float(b['if1_hz']['min']), float(b['if1_hz']['max'])
        rf_min, rf_max = float(b['rf_hz']['min']), float(b['rf_hz']['max'])
        
        if1s = np.arange(if1_min, if1_max + 1e5, g['if1_center_step_hz'])
        rfs = np.arange(rf_min, rf_max + 1e5, g['rf_center_step_hz'])
        bws = g['bw_grid_hz']
        
        tid = 0
        for bw in bws:
            for if1 in if1s:
                if (if1 - bw/2) < if1_min or (if1 + bw/2) > if1_max:
                    continue
                for rf in rfs:
                    self.tiles.append(Tile(tid, if1, bw, rf))
                    tid += 1

    def _build_filter_profile(self, file_path, is_rf=True, strict=False):
        freqs, atten = np.array([]), np.array([])
        loaded = False
        
        if file_path and os.path.exists(file_path):
            try:
                freqs, vals = self._parse_touchstone(file_path)
                if len(freqs) > 0:
                    atten = -vals if np.mean(vals) < 0 else vals
                    loaded = True
            except Exception as e:
                if strict: raise ValueError(f"Failed to parse filter file '{file_path}': {e}")
                print(f"Warning: Failed to load filter {file_path}: {e}")
        elif strict and file_path:
             raise FileNotFoundError(f"Filter file required but not found: {file_path}")

        # Fallback for RF
        if not loaded and is_rf:
            if strict and file_path: raise ValueError("RF Filter load failed in strict mode.")
            b = self.yaml_data['bands']['rf_hz']
            f_min, f_max = b['min'], b['max']
            margin = 0.05 * (f_max - f_min)
            
            freqs = np.array([
                0, f_min - 2*margin, f_min, f_max, f_max + 2*margin, 100e9
            ])
            atten = np.array([80, 80, 1.5, 1.5, 80, 80])
            
        if is_rf: 
            self.rf_filter_raw_freqs, self.rf_filter_raw_atten = freqs, atten
        elif loaded:
            self._normalize_prototype(freqs, atten, strict=strict)

    def _normalize_prototype(self, freqs, atten, strict: bool = False):
        if len(freqs) == 0:
            raise ValueError("S2P file data is empty")

        method = self.yaml_data.get('runtime_settings', {}).get(
            's2p_normalization_method', '3db_width'
        )
        if method != '3db_width':
            print(f"Warning: s2p_normalization_method='{method}' "
                  f"is not implemented. Falling back to '3db_width'.")

        min_idx = np.argmin(atten)
        f_center = freqs[min_idx]
        min_loss = atten[min_idx]
        
        target_loss = min_loss + 3.0
        
        f_lower = f_center
        for i in range(min_idx, -1, -1):
            if atten[i] >= target_loss:
                f_lower = freqs[i]
                break
                
        f_upper = f_center
        for i in range(min_idx, len(freqs)):
            if atten[i] >= target_loss:
                f_upper = freqs[i]
                break
        
        bw_meas = f_upper - f_lower
        
        if bw_meas <= 0 or bw_meas < (freqs[-1] - freqs[0]) * 0.001:
            msg = ("Auto-detected BW from S2P seems invalid. "
                   "Check S2P quality or normalization method.")
            if strict:
                raise ValueError(msg)
            print("Warning:", msg, "Defaulting to unit-width normalization.")
            bw_meas = 1.0
            if f_center <= 0:
                f_center = (freqs[-1] + freqs[0]) / 2.0
        
        self.if2_proto_norm_x = (freqs - f_center) / bw_meas
        self.if2_proto_val_y = atten

    def _parse_touchstone(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        freqs, s21_db = [], []
        scale_factor = 1.0
        units = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
        
        with open(file_path, 'r') as f:
            for line in f:
                clean_line = line.split('!')[0].strip().upper()
                if not clean_line: continue
                
                if clean_line.startswith('#'):
                    parts = clean_line.split()
                    for u, factor in units.items():
                        if u in parts:
                            scale_factor = factor
                            break
                    continue
                    
                if clean_line.startswith("VAR") or clean_line.startswith("BEGIN"): continue
                
                try:
                    parts = clean_line.replace(',', ' ').split()
                    if len(parts) < 2: continue
                    f_val = float(parts[0])
                    val = 0.0
                    if len(parts) == 2: val = float(parts[1]) 
                    elif len(parts) == 3: val = float(parts[1]) 
                    elif len(parts) >= 9: val = float(parts[3]) 
                    else: val = float(parts[1])
                    freqs.append(f_val * scale_factor)
                    s21_db.append(val)
                except ValueError: continue
        
        freqs = np.array(freqs)
        if scale_factor == 1.0 and len(freqs) > 0 and np.max(freqs) < 200.0:
             freqs *= 1e9
        return freqs, np.array(s21_db)
