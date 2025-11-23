import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    # rather than looking like 0 dB or skewing the scale
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
        aggfunc='mean' # Handle duplicates if any by averaging
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
    
    # Create Heatmap
    im = plt.imshow(
        pivot.to_numpy(), 
        aspect='auto', 
        origin='lower', 
        cmap='RdYlGn', 
        vmin=0.0, 
        vmax=1.0,
        extent=[if1_axis.min(), if1_axis.max(), rf_axis.min(), rf_axis.max()]
    )
    
    cbar = plt.colorbar(im)
    cbar.set_label('Spur Margin (dB)')
    
    plt.xlabel('IF1 Center Frequency (MHz)')
    plt.ylabel('RF Center Frequency (GHz)')
    plt.title('BUC Spur Margin Heatmap\n(Green = Good, Red = Fail, White = No Solution)')
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Heatmap saved to {filename}")
    plt.close()