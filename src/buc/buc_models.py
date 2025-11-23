import numpy as np
from .buc_structures import MixerModel, SynthesizerModel

class HardwareStack:
    def __init__(self, cfg, stack_def):
        self.cfg = cfg
        self.name = stack_def['name']
        self.mixer1 = self._find_mixer(cfg, stack_def['mixer1'])
        self.mixer2 = self._find_mixer(cfg, stack_def['mixer2'])
        self.lo1_def = self._find_lo(cfg, stack_def['lo1'], "lo1")
        self.lo2_def = self._find_lo(cfg, stack_def['lo2'], "lo2")
        self.rf_bpf_id = stack_def.get('rf_bpf_file', 'rf_bpf_synthetic')
    
    def _find_mixer(self, cfg, name):
        for m in cfg.yaml_data['mixers']:
            if m['name'] == name:
                derate = m.get('drive_derate', {})
                nom_dbm = derate.get('nominal_dbm', 13.0)
                
                fam_scale = m.get('lo_family_scaling', {})
                slope = fam_scale.get('default_slope_db_per_db', 1.0)
                cap = fam_scale.get('cap_db', 12.0)

                include_iso = m.get('include_isolation_spurs', True)

                return MixerModel(
                    name=m['name'],
                    lo_range=tuple(m['lo_range_hz']),
                    drive_req=(m['required_lo_drive_dbm']['min'], m['required_lo_drive_dbm']['max']),
                    isolation=m.get('isolation', {}),
                    spur_table_raw=[(x[0], x[1], x[2]) for x in m['spur_list']],
                    nom_drive_dbm=nom_dbm,
                    scaling_slope=slope,
                    scaling_cap=cap,
                    include_isolation_spurs=include_iso
                )
        raise ValueError(f"Mixer {name} not found")
    

    def _find_lo(self, cfg, name, role):
        for l in cfg.yaml_data['los']:
            if l['name'] != name:
                continue

            dist_cfg = l.get('distribution', {})
            dist_loss = float(dist_cfg.get('path_losses_db', {}).get(role, 3.0))
            pads = [float(p) for p in dist_cfg.get('pad_options_db', [0, 3, 6])]

            # Output power vs frequency
            p_freqs = np.array([1e6, 40e9], dtype=float)
            p_vals = np.array([float(l.get('output_power_dbm', 0.0))] * 2, dtype=float)

            if 'output_power_model' in l and 'table' in l['output_power_model']:
                tbl = l['output_power_model']['table']
                p_freqs = np.array([float(f) for f in tbl['freq_hz']], dtype=float)
                p_vals = np.array([float(v) for v in tbl['p_out_dbm']], dtype=float)

            modes = l.get('modes', [])
            harmonics = []
            pfd_spurs = []
            # defaults
            pfd_hz = 100e6
            pfd_min, pfd_max = 0.0, 0.0
            frac_en = False
            frac_lvl = -60.0
            frac_slope = 10.0

            lock_base = 0.4
            lock_slope = 0.002

            mode_name = "fracN"
            vco_divs = [1]
            pfd_divs = [1]
            int_frac_penalty = 0.0

            if modes:
                m = modes[0]  # first mode only for now
                mode_name = m.get('name', 'fracN')
                vco_divs = [int(x) for x in m.get('vco_dividers', [1])]
                pfd_divs = [int(x) for x in m.get('pfd_dividers', [1])]

                harmonics = []
                for h in m.get('harmonics', []):
                    # ensure numeric
                    harmonics.append({
                        'k': int(h.get('k', 1)),
                        'rel_dBc': float(h.get('rel_dBc', -60.0)),
                    })

                pfd_cfg = m.get('pfd_spurs_at_output', {})
                if 'families' in pfd_cfg:
                    for fam in pfd_cfg['families']:
                        if 'components' in fam:
                            for comp in fam['components']:
                                pfd_spurs.append({
                                    'k': int(comp.get('k', 1)),
                                    'base_rel_dBc': float(comp.get('base_rel_dBc', -60.0)),
                                    'rolloff_dB_per_dec': float(comp.get('rolloff_dB_per_dec', 0.0)),
                                })

                if 'pfd_hz_range' in m:
                    pfd_min = float(m['pfd_hz_range'][0])
                    pfd_max = float(m['pfd_hz_range'][1])
                    pfd_hz = (pfd_min + pfd_max) / 2.0

                fb = m.get('frac_boundary_spurs', {})
                frac_en = bool(fb.get('enabled', False))
                frac_lvl = float(fb.get('amplitude_at_eps0p5_rel_dBc', -58.0))
                frac_slope = float(fb.get('rolloff_slope_db_per_dec', 10.0))

                lt = m.get('lock_time_model', {})
                lock_base = float(lt.get('base_ms', 0.4))
                lock_slope = float(lt.get('per_mhz_ms', 0.002))
                penalties = lt.get('mode_penalties_ms', {})
                int_frac_penalty = float(penalties.get('int_to_frac', 0.0))

            div_spec_raw = l.get('divider_spectrum', {})
            # normalize divider_spectrum to use floats
            div_spec = {}
            for key, val in div_spec_raw.items():
                try:
                    div_spec[key] = {
                        'harm_delta_dBc': float(val.get('harm_delta_dBc', 0.0))
                    }
                except Exception:
                    # if malformed, just skip this entry
                    continue

            return SynthesizerModel(
                name=l['name'],
                freq_range=(float(l['freq_range_hz'][0]), float(l['freq_range_hz'][1])),
                step_hz=float(l.get('step_hz', 100e3)),
                power_freqs=p_freqs,
                power_dbm=p_vals,
                dist_loss_db=dist_loss,
                pad_options=pads,
                harmonics=harmonics,
                pfd_freq_hz=pfd_hz,
                pfd_spurs=pfd_spurs,
                frac_boundary_enabled=frac_en,
                frac_boundary_lvl=frac_lvl,
                frac_boundary_slope=frac_slope,
                pfd_min_hz=pfd_min,
                pfd_max_hz=pfd_max,
                lock_base_ms=lock_base,
                lock_per_mhz_ms=lock_slope,
                vco_dividers=vco_divs,
                pfd_dividers=pfd_divs,
                mode_name=mode_name,
                int_frac_switch_penalty_ms=int_frac_penalty,
                divider_spectrum=div_spec,
            )
        raise ValueError(f"LO {name} not found")

    def get_valid_lo_config(self, lo_model: SynthesizerModel, target_freq, mixer_drive_req):
        if target_freq < lo_model.freq_range[0] or target_freq > lo_model.freq_range[1]:
            return False, 0, 0

        proj = getattr(self, "cfg", None)
        if proj is not None and lo_model.pfd_min_hz > 0.0 and lo_model.pfd_max_hz > 0.0:
            ref = float(proj.yaml_data.get('project', {}).get('reference_10mhz_hz', 0.0))
            if ref > 0.0:
                f_pfd_min = ref / 16.0
                f_pfd_max = ref
                if lo_model.pfd_max_hz < f_pfd_min or lo_model.pfd_min_hz > f_pfd_max:
                    if not hasattr(self, "_pfd_warned"):
                        self._pfd_warned = set()
                    key = lo_model.name
                    if key not in self._pfd_warned:
                        print(f"Warning: LO '{lo_model.name}' pfd_hz_range "
                              f"{lo_model.pfd_min_hz/1e6:.1f}–{lo_model.pfd_max_hz/1e6:.1f} MHz "
                              f"does not overlap ref-based window {f_pfd_min/1e6:.3f}–{f_pfd_max/1e6:.3f} MHz.")
                        self._pfd_warned.add(key)

        p_source = np.interp(target_freq, lo_model.power_freqs, lo_model.power_dbm)
        
        valid_pads = []
        for pad in lo_model.pad_options:
            p_del = p_source - lo_model.dist_loss_db - pad
            if mixer_drive_req[0] <= p_del <= mixer_drive_req[1]:
                valid_pads.append((pad, p_del))
        
        if not valid_pads:
            return False, 0, 0
        
        valid_pads.sort(key=lambda x: (x[0], x[1]), reverse=True) 
        best = valid_pads[0]
        return True, best[0], best[1]

    def generate_lo_spectrum(self, lo_model: SynthesizerModel, f_center: float, f_pfd: float = None):
        """
        Generates 2D numpy array [[freq, rel_dBc], ...].
        Now accepts explicit f_pfd and applies divider spectrum deltas.
        """
        if f_pfd is None:
            f_pfd = lo_model.pfd_freq_hz

        comps = []
        # Main LO Tone
        comps.append([f_center, 0.0])
        
        # Harmonics with divider-dependent adjustment
        # Phase-1 assumption: operating effectively at /1 for now, 
        # but scaffolding allows future variable dividers.
        div_key = "/1" 
        harm_delta = 0.0
        if lo_model.divider_spectrum:
            harm_delta = float(lo_model.divider_spectrum.get(div_key, {}).get("harm_delta_dBc", 0.0))
        
        for h in lo_model.harmonics:
            lvl = h['rel_dBc'] + harm_delta
            comps.append([f_center * h['k'], lvl])
            
        # PFD Spurs
        # Note: detailed rolloff_dB_per_dec is parsed but currently ignored in Phase-1
        for p in lo_model.pfd_spurs:
            offset = p['k'] * f_pfd
            base_lvl = p.get('base_rel_dBc', -60.0)
            lvl = base_lvl
            comps.append([f_center + offset, lvl])
            comps.append([f_center - offset, lvl])

        # Fractional boundary spurs
        if lo_model.frac_boundary_enabled and f_pfd > 0:
            N_float = f_center / f_pfd
            N_int = round(N_float)
            epsilon = abs(N_float - N_int)
            
            if epsilon > 1e-6:
                f_boundary = N_int * f_pfd
                if epsilon > 0.5: epsilon = 1.0 - epsilon 
                
                delta_dec = np.log10(0.5 / epsilon)
                lvl = lo_model.frac_boundary_lvl + lo_model.frac_boundary_slope * delta_dec
                
                lvl = min(0, lvl)
                comps.append([f_boundary, lvl])

        return np.array(comps, dtype=np.float64)