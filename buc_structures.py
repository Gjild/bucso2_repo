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
class MixerModel:
    name: str
    lo_range: Tuple[float, float]
    drive_req: Tuple[float, float]
    isolation: Dict[str, float]
    spur_table_raw: List[Tuple[int, int, float]]
    # Flattened numpy array for JIT: [[m, n, rej], ...]
    spur_table_np: np.ndarray = field(init=False)

    def __post_init__(self):
        self.spur_table_np = np.array(self.spur_table_raw, dtype=np.float64)

@dataclass
class SynthesizerModel:
    name: str
    freq_range: Tuple[float, float]
    step_hz: float
    # Power Model
    power_freqs: np.ndarray
    power_dbm: np.ndarray
    # Distribution
    dist_loss_db: float
    pad_options: List[float]
    # Spectral Purity
    harmonics: List[Dict] 
    pfd_freq_hz: float
    pfd_spurs: List[Dict]
    # Fractional Boundary Spurs
    frac_boundary_enabled: bool
    frac_boundary_lvl: float  # Level at epsilon=0.5
    frac_boundary_slope: float

@dataclass
class LO_Candidate:
    freq_hz: float
    mode: str 
    side: str 
    delivered_power_dbm: float
    pad_db: float
    # 2D Array [[freq_hz, rel_dBc], ...] - Contains Main + Harmonics + Spurs
    spectrum: np.ndarray 

@dataclass
class FilterModel:
    center_hz: float = 0.0
    bw_hz: float = 0.0
    model_type: int = 0  # 0=Symmetric, 1=S2P
    passband_il: float = 1.0
    rolloff: float = 40.0 
    stop_floor: float = -80.0
    # Pointers to global prototype arrays (not stored per instance to save RAM)
    proto_idx: int = -1 

@dataclass
class GlobalConfig:
    yaml_data: dict
    tiles: List[Tile] = field(default_factory=list)
    
    # Global Settings
    grid_max_freq_hz: float = 40e9
    grid_step_hz: float = 1e6
    opt_cutoff_db: float = -20.0
    noise_floor_dbc: float = -100.0
    
    # Pre-loaded physics Arrays
    rf_filter_raw_freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    rf_filter_raw_atten: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Normalized IF2 Prototype (X axis = normalized frequency diff, Y = dB)
    if2_proto_norm_x: np.ndarray = field(default_factory=lambda: np.array([]))
    if2_proto_val_y: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        cfg = GlobalConfig(yaml_data=data)
        cfg._generate_tiles()
        
        # Load Grid Config
        grid_cfg = data.get('rbw_binning', {})
        grid_glob = data.get('grids', {})
        cfg.grid_step_hz = float(grid_cfg.get('lut_step_hz', 1e6)) 
        cfg.grid_max_freq_hz = float(grid_glob.get('grid_max_freq_hz', 45e9))
        
        # Load Settings
        run_set = data.get('runtime_settings', {})
        cfg.opt_cutoff_db = float(run_set.get('optimization_cutoff_db', -20.0))
        cfg.noise_floor_dbc = float(run_set.get('noise_floor_cutoff_dbc', -100.0))
        
        hw_choices = data.get('hardware_choices', {}).get('stacks', [])
        rf_file = hw_choices[0].get('rf_bpf_file', None) if hw_choices else None
        cfg._build_filter_profile(rf_file, is_rf=True)
        
        if2_opts = data.get('if2_model', {}).get('scaled_s2p', {})
        if if2_opts.get('enabled', False):
            cfg._build_filter_profile(if2_opts.get('prototype_s2p_file'), is_rf=False)
            
        return cfg

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

    def _build_filter_profile(self, file_path, is_rf=True):
        freqs, atten = np.array([]), np.array([])
        loaded = False
        
        if file_path and os.path.exists(file_path):
            try:
                # Basic S2P/CSV parser: Freq(Hz or GHz), S21(dB)
                data = np.loadtxt(file_path, comments=['!', '#'], delimiter=None)
                if data.shape[0] > 1 and data.shape[1] >= 2:
                    freqs = data[:, 0]
                    # Robust GHz detection: if max freq is small, assume GHz
                    if np.max(freqs) < 200.0: 
                        freqs *= 1e9 
                    vals = data[:, 1]
                    # Ensure attenuation is positive dB (Loss)
                    atten = -vals if np.mean(vals) < 0 else vals
                    loaded = True
            except Exception as e: 
                print(f"Warning: Failed to load filter {file_path}: {e}")

        # Fallback for RF
        if not loaded and is_rf:
            b = self.yaml_data['bands']['rf_hz']
            freqs = np.array([0, b['min']-1e6, b['min'], b['max'], b['max']+1e6, 100e9])
            atten = np.array([80, 80, 1.0, 1.0, 80, 80])
            
        if is_rf: 
            self.rf_filter_raw_freqs, self.rf_filter_raw_atten = freqs, atten
        elif loaded:
            # Normalize IF2 Prototype
            # 1. Find Center (Minimum Loss)
            min_idx = np.argmin(atten)
            f_center = freqs[min_idx]
            min_loss = atten[min_idx]
            
            # 2. Find 3dB BW
            target_loss = min_loss + 3.0
            # Simple search
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
            if bw_meas <= 0: bw_meas = 1.0 # Safety
            
            # 3. Normalize
            # X = (f - f_c) / BW
            # Y = atten
            self.if2_proto_norm_x = (freqs - f_center) / bw_meas
            self.if2_proto_val_y = atten