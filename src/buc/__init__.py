# src/buc/__init__.py

from .buc_structures import (
    GlobalConfig,
    Tile,
    FilterModel,
    MixerModel,
    SynthesizerModel,
    LO_Candidate,
    TilePolicyEntry,
)

from .buc_models import HardwareStack
from .buc_engine import SpurEngine
from .buc_diagnostics import generate_spur_ledger
from .buc_visuals import plot_margin_heatmap
from .buc_markov import markov_lock_summary
from .buc_validation import basic_validation

from .buc_kernels import (
    build_dense_lut,
    fill_symmetric_filter_lut,
    precompute_mixing_recipes,
    compute_stage1_spurs_no_if2,
    compute_stage2_from_intermediates,
)

__all__ = [
    "GlobalConfig",
    "Tile",
    "FilterModel",
    "MixerModel",
    "SynthesizerModel",
    "LO_Candidate",
    "TilePolicyEntry",
    "HardwareStack",
    "SpurEngine",
    "generate_spur_ledger",
    "plot_margin_heatmap",
    "markov_lock_summary",
    "basic_validation",
    "build_dense_lut",
    "fill_symmetric_filter_lut",
    "precompute_mixing_recipes",
    "compute_stage1_spurs_no_if2",
    "compute_stage2_from_intermediates",
]
