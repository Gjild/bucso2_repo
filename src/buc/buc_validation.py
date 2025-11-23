from .buc_engine import SpurEngine
from .buc_models import HardwareStack

def basic_validation(cfg, stack_def, if2_filter):
    """
    Basic validation per TDS §14 (coarse vs full-order sparse):
    - Compares search-mode vs full-order sparse margins for a few tiles.
    Prints warning if divergence exceeds a small tolerance.

    NOTE: This is not a dense per-tone simulation; it's 'coarse vs fine
    sparse' rather than 'sparse vs dense' as in the full TDS.
    """
    hw = HardwareStack(cfg, stack_def)
    engine = SpurEngine(cfg, hw)
    
    if not cfg.tiles:
        print("No tiles to validate.")
        return
        
    tiles_to_check = cfg.tiles[:3]  # first few tiles
    tol_db = 0.5
    
    print("Running Basic Validation (Search Mode vs Full Physics)...")
    for t in tiles_to_check:
        m_sum_search = engine._eval_chain_fast(t, if2_filter, high_side=False, search_mode=True)
        m_sum_full = engine._eval_chain_fast(t, if2_filter, high_side=False, search_mode=False)
        
        m_diff_search = engine._eval_chain_fast(t, if2_filter, high_side=True, search_mode=True)
        m_diff_full = engine._eval_chain_fast(t, if2_filter, high_side=True, search_mode=False)
        
        for label, ms, mf in (
            ("Sum", m_sum_search, m_sum_full),
            ("Diff", m_diff_search, m_diff_full),
        ):
            if ms <= -900 or mf <= -900:
                continue
            if abs(ms - mf) > tol_db:
                print(f"Validation warning: Tile {t.id}, {label} path "
                      f"search/full mismatch {ms:.2f} vs {mf:.2f} dB.")
    print("Validation Complete.")