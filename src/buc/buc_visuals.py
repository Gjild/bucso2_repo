import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap  # <- NEW

def plot_margin_heatmap(df_policy: pd.DataFrame, filename: str = "heatmap_margin.png"):
    """
    Generates a 2D heatmap of Spur Margin vs IF1 and RF Center frequencies.
    """
    if df_policy.empty:
        print("No policy data to plot.")
        return

    # Work on a copy to avoid modifying the original df passed in memory
    df = df_policy.copy()

    # Validation: Ensure we have the required columns for axes
    required_cols = {'if1_center_hz', 'rf_center_hz'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: Missing required columns for heatmap axes: {missing}")
        return

    # Normalization: Ensure target column exists
    if 'spur_margin_db' not in df.columns and 'margin' in df.columns:
        df['spur_margin_db'] = df['margin']
    
    if 'spur_margin_db' not in df.columns:
        print("Error: 'spur_margin_db' column missing from policy dataframe.")
        return

    # Handle sentinel "no-solution" margins (e.g. -999.0) as NaN so they stand out
    margins = df["spur_margin_db"].astype(float).copy()
    sentinel_mask = margins <= -800.0
    margins[sentinel_mask] = np.nan
    df["spur_margin_db"] = margins

    # Pivot Data
    # X-axis: IF1, Y-axis: RF, Value: Margin
    pivot = df.pivot_table(
        index='rf_center_hz', 
        columns='if1_center_hz', 
        values='spur_margin_db',
        aggfunc='mean'  # Handle duplicates if any by averaging
    )

    if pivot.empty:
        print("No pivot data to plot.")
        return
    
    # Ensure sorted axes
    pivot = pivot.sort_index().sort_index(axis=1)
    
    # Convert index/cols for readability
    rf_axis = pivot.index.to_numpy() / 1e9     # RF in GHz
    if1_axis = pivot.columns.to_numpy() / 1e6  # IF1 in MHz
    
    plt.figure(figsize=(12, 8))

    # ------------------------------------------------------------------
    # Custom colormap:
    # - For margins < 0: light red (near 0 dB) to dark red (−5 dB)
    # - For margins > 0: light green (near 0 dB) to dark green (+5 dB)
    # - 0 is a hard color boundary between red and green
    # - NaNs (no-solution) will be white
    # ------------------------------------------------------------------
    dark_red   = (0.6, 0.0, 0.0)   # stronger fail
    light_red  = (1.0, 0.8, 0.8)   # just-barely fail
    light_green = (0.8, 1.0, 0.8)  # just-barely pass
    dark_green  = (0.0, 0.4, 0.0)  # strong pass

    cdict = {
        # x in [0, 1] corresponds to margin in [-5, +5] via vmin/vmax
        # x = 0   -> -5 dB (dark_red)
        # x = 0.5 ->  0 dB (transition from light_red to light_green)
        # x = 1   -> +5 dB (dark_green)
        'red': (
            (0.0, dark_red[0],   dark_red[0]),
            (0.5, light_red[0],  light_green[0]),
            (1.0, dark_green[0], dark_green[0]),
        ),
        'green': (
            (0.0, dark_red[1],   dark_red[1]),
            (0.5, light_red[1],  light_green[1]),
            (1.0, dark_green[1], dark_green[1]),
        ),
        'blue': (
            (0.0, dark_red[2],   dark_red[2]),
            (0.5, light_red[2],  light_green[2]),
            (1.0, dark_green[2], dark_green[2]),
        ),
    }

    margin_cmap = LinearSegmentedColormap('FailPassMargin', cdict, N=256)
    margin_cmap.set_bad('white')  # keep NaNs as white = "no solution"
    # ------------------------------------------------------------------

    # Create Heatmap
    im = plt.imshow(
        pivot.to_numpy(), 
        aspect='auto', 
        origin='lower', 
        cmap=margin_cmap,  # <- use our custom colormap
        vmin=-5.0, 
        vmax=5.0,
        extent=[if1_axis.min(), if1_axis.max(), rf_axis.min(), rf_axis.max()]
    )
    
    cbar = plt.colorbar(im)
    cbar.set_label('Spur Margin (dB)')
    
    plt.xlabel('IF1 Center Frequency (MHz)')
    plt.ylabel('RF Center Frequency (GHz)')
    plt.title('BUC Spur Margin Heatmap\n(Green = Pass, Red = Fail, White = No Solution)')
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Heatmap saved to {filename}")
    plt.close()


def plot_if2_filter(engine, if2_filter, csv_path: str = "if2_filter_response.csv",
                    png_path: str = "if2_filter_response.png"):
    """
    Export and plot the final IF2 filter attenuation vs frequency.

    Uses the engine's current IF2 LUT (engine.if2_lut_buffer), which must have
    been built for 'if2_filter' via engine.build_policy_for_if2(...) or
    engine.evaluate_policy(...).

    CSV columns: freq_hz, if2_atten_db
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    lut = engine.if2_lut_buffer
    step = engine.grid_step       # Hz per bin
    n_bins = len(lut)

    # Limit plot/export to a sensible window around the filter:
    # same spirit as fill_symmetric_filter_lut -> ±5 * BW around center
    span_hz = 5.0 * if2_filter.bw_hz
    f_start = max(0.0, if2_filter.center_hz - span_hz)
    f_stop  = min(engine.grid_max, if2_filter.center_hz + span_hz)

    idx0 = max(0, int(f_start / step))
    idx1 = min(n_bins, int(f_stop / step) + 1)

    freqs = np.arange(idx0, idx1) * step
    atten = lut[idx0:idx1].copy()

    # --- CSV export ---
    df = pd.DataFrame({
        "freq_hz": freqs,
        "if2_gain_db": -atten,
    })
    df.to_csv(csv_path, index=False)
    print(f"IF2 filter response CSV written to {csv_path}")

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e9, -atten)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("IF2 Gain (dB)")
    plt.title(
        f"IF2 Filter Response\n"
        f"center = {if2_filter.center_hz/1e9:.3f} GHz, "
        f"BW = {if2_filter.bw_hz/1e6:.1f} MHz"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"IF2 filter response plot saved to {png_path}")
    plt.close()
