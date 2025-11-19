import numpy as np
from buc_structures import MixerModel, SynthesizerModel

class HardwareStack:
    def __init__(self, cfg, stack_def):
        self.name = stack_def['name']
        self.mixer1 = self._find_mixer(cfg, stack_def['mixer1'])
        self.mixer2 = self._find_mixer(cfg, stack_def['mixer2'])
        self.lo1_def = self._find_lo(cfg, stack_def['lo1'], "lo1")
        self.lo2_def = self._find_lo(cfg, stack_def['lo2'], "lo2")
    
    def _find_mixer(self, cfg, name):
        for m in cfg.yaml_data['mixers']:
            if m['name'] == name:
                return MixerModel(
                    name=m['name'],
                    lo_range=tuple(m['lo_range_hz']),
                    drive_req=(m['required_lo_drive_dbm']['min'], m['required_lo_drive_dbm']['max']),
                    isolation=m['isolation'],
                    spur_table_raw=[(x[0], x[1], x[2]) for x in m['spur_list']]
                )
        raise ValueError(f"Mixer {name} not found")

    def _find_lo(self, cfg, name, role):
        for l in cfg.yaml_data['los']:
            if l['name'] == name:
                # Distribution & Pads
                dist_cfg = l.get('distribution', {})
                dist_loss = dist_cfg.get('path_losses_db', {}).get(role, 3.0)
                pads = dist_cfg.get('pad_options_db', [0, 3, 6])

                # Power Model
                p_freqs = np.array([1e6, 40e9])
                p_vals = np.array([l.get('output_power_dbm', 0)] * 2)
                
                if 'output_power_model' in l and 'table' in l['output_power_model']:
                    tbl = l['output_power_model']['table']
                    p_freqs = np.array(tbl['freq_hz'])
                    p_vals = np.array(tbl['p_out_dbm'])

                # Spectral Defs
                modes = l.get('modes', [])
                harmonics = []
                pfd_spurs = []
                pfd_hz = 100e6
                
                frac_en = False
                frac_lvl = -60
                frac_slope = 10
                
                if modes:
                    m = modes[0] # Default to first mode
                    harmonics = m.get('harmonics', [])
                    
                    # PFD Spurs
                    pfd_cfg = m.get('pfd_spurs_at_output', {})
                    if 'families' in pfd_cfg:
                        for fam in pfd_cfg['families']:
                             if 'components' in fam:
                                 pfd_spurs.extend(fam['components'])
                    
                    if 'pfd_hz_range' in m:
                        pfd_hz = (m['pfd_hz_range'][0] + m['pfd_hz_range'][1]) / 2
                    
                    # Fractional Boundary
                    fb = m.get('frac_boundary_spurs', {})
                    frac_en = fb.get('enabled', False)
                    frac_lvl = fb.get('amplitude_at_eps0p5_rel_dBc', -58)
                    frac_slope = fb.get('rolloff_slope_db_per_dec', 10)

                return SynthesizerModel(
                    name=l['name'],
                    freq_range=tuple(l['freq_range_hz']),
                    step_hz=l.get('step_hz', 100e3),
                    power_freqs=p_freqs,
                    power_dbm=p_vals,
                    dist_loss_db=dist_loss,
                    pad_options=pads,
                    harmonics=harmonics,
                    pfd_freq_hz=pfd_hz,
                    pfd_spurs=pfd_spurs,
                    frac_boundary_enabled=frac_en,
                    frac_boundary_lvl=frac_lvl,
                    frac_boundary_slope=frac_slope
                )
        raise ValueError(f"LO {name} not found")

    def get_valid_lo_config(self, lo_model: SynthesizerModel, target_freq, mixer_drive_req):
        """
        Checks range, and pads. Returns (is_valid, pad_used, delivered_power).
        """
        if target_freq < lo_model.freq_range[0] or target_freq > lo_model.freq_range[1]:
            return False, 0, 0

        p_source = np.interp(target_freq, lo_model.power_freqs, lo_model.power_dbm)
        
        valid_pads = []
        for pad in lo_model.pad_options:
            p_del = p_source - lo_model.dist_loss_db - pad
            if mixer_drive_req[0] <= p_del <= mixer_drive_req[1]:
                valid_pads.append((pad, p_del))
        
        if not valid_pads:
            return False, 0, 0
        
        # Prefer lowest power that meets requirement to save energy/compression
        valid_pads.sort(key=lambda x: x[1]) 
        best = valid_pads[0]
        return True, best[0], best[1]

    def generate_lo_spectrum(self, lo_model: SynthesizerModel, f_center):
        """
        Generates 2D numpy array [[freq, rel_dBc], ...].
        Includes Main Tone, Harmonics, PFD spurs, and Boundary Spurs.
        """
        comps = []
        # 1. Main Tone
        comps.append([f_center, 0.0])
        
        # 2. Harmonics
        for h in lo_model.harmonics:
            comps.append([f_center * h['k'], h['rel_dBc']])
            
        # 3. PFD Spurs
        for p in lo_model.pfd_spurs:
            offset = p['k'] * lo_model.pfd_freq_hz
            lvl = p.get('base_rel_dBc', -60)
            comps.append([f_center + offset, lvl])
            comps.append([f_center - offset, lvl])

        # 4. Fractional Boundary Spurs
        if lo_model.frac_boundary_enabled and lo_model.pfd_freq_hz > 0:
            # Calculate N and fraction
            N_float = f_center / lo_model.pfd_freq_hz
            N_int = round(N_float)
            
            # Distance to integer boundary (epsilon)
            epsilon = abs(N_float - N_int)
            
            # If exactly integer, no boundary spur (or handling integer mode)
            if epsilon > 1e-6:
                # Boundary spur frequency is at distance (epsilon * f_pfd) from carrier
                # which maps physically to: f_spur = f_center +/- (epsilon * f_pfd)
                # But wait, boundary spurs usually appear at N_int * f_pfd
                # f_boundary = N_int * f_pfd
                f_boundary = N_int * lo_model.pfd_freq_hz
                
                # Amplitude model: L = L_0.5 + Slope * log10(0.5 / epsilon)
                # Smaller epsilon (closer to boundary) -> Higher spur
                if epsilon > 0.5: epsilon = 1.0 - epsilon # normalize 0..0.5
                
                delta_dec = np.log10(0.5 / epsilon)
                lvl = lo_model.frac_boundary_lvl + lo_model.frac_boundary_slope * delta_dec
                
                # Cap at 0 dBc just in case
                lvl = min(0, lvl)
                comps.append([f_boundary, lvl])

        return np.array(comps, dtype=np.float64)