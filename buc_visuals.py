import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_margin_heatmap(df_policy, filename="heatmap_margin.png"):
    """
    Generates a 2D heatmap of Spur Margin vs IF1 and RF Center frequencies.
    """
    if df_policy.empty:
        print("No policy data to plot.")
        return

    # Pivot Data
    # X-axis: IF1 (MHz), Y-axis: RF (GHz), Value: Margin
    pivot = df_policy.pivot_table(
        index='rf_hz', 
        columns='if1_hz', 
        values='margin'
    )
    
    # Convert index/cols for readability
    pivot.index = pivot.index / 1e9
    pivot.columns = pivot.columns / 1e6
    
    plt.figure(figsize=(12, 8))
    
    # Create Heatmap
    # Vmin/Vmax clamp color scale to useful range (e.g., 0dB to 20dB)
    plt.imshow(
        pivot, 
        aspect='auto', 
        origin='lower', 
        cmap='RdYlGn', 
        vmin=0, 
        vmax=20,
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()]
    )
    
    cbar = plt.colorbar()
    cbar.set_label('Spur Margin (dB)')
    
    plt.xlabel('IF1 Center Frequency (MHz)')
    plt.ylabel('RF Center Frequency (GHz)')
    plt.title('BUC Spur Margin Heatmap\n(Green = Good, Red = Fail)')
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Heatmap saved to {filename}")
    plt.close()