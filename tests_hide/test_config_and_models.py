# tests/test_config_and_models.py
from pathlib import Path

import numpy as np
import pytest
import yaml

from buc import GlobalConfig, MixerModel
from buc import HardwareStack


def test_global_config_tiles_respect_if1_and_rf_bands(cfg: GlobalConfig):
    """
    Ensure tile generation respects IF1 band edges and BW containment.
    """
    assert cfg.tiles, "Expected some tiles to be generated"

    b_if = cfg.yaml_data["bands"]["if1_hz"]
    b_rf = cfg.yaml_data["bands"]["rf_hz"]
    if1_min, if1_max = float(b_if["min"]), float(b_if["max"])
    rf_min, rf_max = float(b_rf["min"]), float(b_rf["max"])

    for t in cfg.tiles:
        lo_if = t.if1_center_hz - t.bw_hz / 2.0
        hi_if = t.if1_center_hz + t.bw_hz / 2.0
        assert if1_min <= lo_if <= if1_max
        assert if1_min <= hi_if <= if1_max
        assert rf_min <= t.rf_center_hz <= rf_max


def test_rf_filter_fallback_profile_has_reasonable_passband(cfg: GlobalConfig):
    """
    With rf_bpf_file=null in the shipped config, RF filter response is synthetic.
    We expect:
      - Non-empty frequency & attenuation arrays
      - Attenuation in RF band roughly equal to the configured passband IL (~1.5 dB)
    """
    freqs = cfg.rf_filter_raw_freqs
    atten = cfg.rf_filter_raw_atten
    assert freqs.size > 0
    assert atten.size == freqs.size

    b_rf = cfg.yaml_data["bands"]["rf_hz"]
    rf_min, rf_max = float(b_rf["min"]), float(b_rf["max"])
    mid_rf = 0.5 * (rf_min + rf_max)

    idx_mid = int(np.argmin(np.abs(freqs - mid_rf)))
    att_mid = atten[idx_mid]

    # Synthetic profile uses ~1.5 dB in-pass; just ensure it's not absurd.
    assert 0.0 <= att_mid <= 10.0


def test_mixer_model_keeps_11_entry():
    """
    Regression: MixerModel.__post_init__ must NOT strip the (1,1) entry.
    """
    m = MixerModel(
        name="TEST_MXR",
        lo_range=(1e6, 10e9),
        drive_req=(0.0, 10.0),
        isolation={"lo_to_rf_db": -30.0, "if_to_rf_db": -50.0},
        spur_table_raw=[
            (1, 1, -30.0),
            (1, 2, -40.0),
        ],
        nom_drive_dbm=13.0,
        scaling_slope=1.0,
        scaling_cap=12.0,
        include_isolation_spurs=True,
    )

    assert m.spur_table_np.shape[0] == 2
    assert any((row[0] == 1 and row[1] == 1) for row in m.spur_table_np)


def test_non_11_desired_orders_raise_not_implemented(tmp_path: Path, config_path: Path):
    """
    GlobalConfig.load should raise NotImplementedError when desired_stage*_mn != (1,1).
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Force non-(1,1) desired orders
    data.setdefault("constraints", {})
    data["constraints"]["desired_stage1_mn"] = [2, 1]
    data["constraints"]["desired_stage2_mn"] = [1, 1]

    new_cfg_path = tmp_path / "config_non11.yaml"
    with open(new_cfg_path, "w") as f:
        yaml.safe_dump(data, f)

    with pytest.raises(NotImplementedError):
        GlobalConfig.load(str(new_cfg_path))


def test_hardwarestack_lo_config_respects_mixer_drive(hw_stack: HardwareStack):
    """
    get_valid_lo_config should only return pads that deliver power within
    the mixer's required drive range.
    """
    lo_model = hw_stack.lo1_def
    mixer = hw_stack.mixer1

    target_freq = 0.5 * (lo_model.freq_range[0] + lo_model.freq_range[1])

    valid, pad_db, p_del = hw_stack.get_valid_lo_config(
        lo_model, target_freq, mixer.drive_req
    )

    assert valid
    assert pad_db in lo_model.pad_options
    assert mixer.drive_req[0] <= p_del <= mixer.drive_req[1]
