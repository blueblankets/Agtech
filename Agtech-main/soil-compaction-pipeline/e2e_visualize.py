"""
E2E Heatmap Visualizations — uses lat/lon binning for any pixel count.
Titles now reflect actual data source (live vs synthetic).
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


def e2e_visualize(master_parquet: str, payload_json: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(master_parquet)
    print(f"Loaded {len(df)} pixels from master_df.")

    with open(payload_json, 'r') as f:
        payload = json.load(f)
    df_payload = pd.DataFrame(payload)
    print(f"Loaded {len(df_payload)} payload records.")

    # --- Read manifest for data source transparency ---
    manifest_path = os.path.join(os.path.dirname(master_parquet), "manifest.json")
    data_label = "Pipeline Output"
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        api_health = manifest.get("api_health", {})
        sources = []
        for k, v in api_health.items():
            if "LIVE" in str(v):
                sources.append("LIVE")
            elif "SYNTHETIC" in str(v):
                sources.append("SYNTHETIC")
            elif "SIMULATED" in str(v):
                sources.append("SIM")
        if sources:
            data_label = " + ".join(sorted(set(sources)))

    # --- Sanity Checks ---
    print("\n--- Sanity Checks ---")
    invalid_actions = set(df['action'].unique()) - {
        "Targeted Deep Tillage", "Monitor - Not Economically Viable", "None", "INVALID_DATA"
    }
    print(f"{'FAIL' if invalid_actions else 'PASS'}: Action enum check. Invalid={invalid_actions or 'none'}")

    depth_ok = df['pred_ripper_depth_cm'].dropna().between(0, 60).all()
    print(f"{'PASS' if depth_ok else 'FAIL'}: pred_ripper_depth_cm in [0, 60]")

    stress_ok = df['max_subsoil_stress_mpa'].dropna().between(0, 5).all()
    print(f"{'PASS' if stress_ok else 'FAIL'}: max_subsoil_stress_mpa in [0, 5]")

    roi_range = df['roi'].describe()
    print(f"\nROI stats: min={roi_range['min']:.3f}, max={roi_range['max']:.3f}, mean={roi_range['mean']:.3f}")

    print(f"\nAction distribution:\n{df['action'].value_counts()}")

    # --- Build 2D spatial grid via lat/lon binning ---
    GRID = 40
    df['lat_bin'] = pd.cut(df['lat'], bins=GRID, labels=False)
    df['lon_bin'] = pd.cut(df['lon'], bins=GRID, labels=False)

    # 1. Action Bar Chart
    plt.figure(figsize=(8, 5))
    color_order = {
        'INVALID_DATA': '#94A3B8', 'None': '#16A34A',
        'Monitor - Not Economically Viable': '#D97706',
        'Targeted Deep Tillage': '#DC2626',
    }
    counts = df['action'].value_counts()
    colors = [color_order.get(a, '#94A3B8') for a in counts.index]
    counts.plot(kind='bar', color=colors)
    plt.title(f"Distribution of Recommended Actions ({data_label})")
    plt.ylabel("Pixel Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_actions.png"), dpi=150)
    plt.close()

    # 2. ROI vs Ripper Depth scatter
    valid = df[df['action'] != 'INVALID_DATA'].copy()
    if not valid.empty:
        plt.figure(figsize=(8, 5))
        plt.scatter(valid['roi'], valid['pred_ripper_depth_cm'], alpha=0.4, s=8, c='royalblue')
        plt.axvline(x=1.2, color='r', linestyle='--', label='ROI Trigger (1.2)')
        plt.title(f"ROI vs Predicted Ripper Depth ({data_label})")
        plt.xlabel("Economic ROI")
        plt.ylabel("Predicted Ripper Depth (cm)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_roi_depth.png"), dpi=150)
        plt.close()

    # 3. Numeric Heatmaps
    numeric_cols = [
        ("max_subsoil_stress_mpa", "rocket_r"),
        ("depth_of_max_stress_cm", "YlOrRd"),
        ("pred_ripper_depth_cm", "YlGnBu"),
        ("mapie_lower_bound", "Blues"),
        ("mapie_upper_bound", "Oranges"),
        ("roi", "coolwarm"),
    ]
    for col, cmap in numeric_cols:
        if col not in df.columns:
            continue
        plt.figure(figsize=(10, 8))
        pivot = df.pivot_table(index='lat_bin', columns='lon_bin', values=col, aggfunc='mean')
        pivot = pivot.sort_index(ascending=False)
        sns.heatmap(pivot, cmap=cmap, annot=False, xticklabels=False, yticklabels=False)
        plt.title(f"Heatmap: {col} ({data_label})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_{col}.png"), dpi=150)
        plt.close()

    # 4. Action Prescription Map
    action_map = {
        "Targeted Deep Tillage": 3,
        "Monitor - Not Economically Viable": 2,
        "None": 1,
        "INVALID_DATA": 0,
    }
    df['action_code'] = df['action'].map(action_map).fillna(0)
    plt.figure(figsize=(10, 8))
    pivot_action = df.pivot_table(index='lat_bin', columns='lon_bin', values='action_code', aggfunc='max')
    pivot_action = pivot_action.sort_index(ascending=False)
    cmap_action = sns.color_palette(["#94A3B8", "#16A34A", "#D97706", "#DC2626"])
    sns.heatmap(pivot_action, cmap=cmap_action, annot=False, cbar=False,
                xticklabels=False, yticklabels=False)
    plt.title(f"Prescription Action Map ({data_label})")
    legend_elements = [
        Patch(facecolor='#DC2626', label='Targeted Deep Tillage'),
        Patch(facecolor='#D97706', label='Monitor'),
        Patch(facecolor='#16A34A', label='None'),
        Patch(facecolor='#94A3B8', label='INVALID_DATA'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_action.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved all plots to {output_dir}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    e2e_visualize(
        os.path.join(base, "pipeline_data", "master_df.parquet"),
        os.path.join(base, "pipeline_data", "final_payload.json"),
        os.path.join(base, "e2e_visualizations"),
    )
