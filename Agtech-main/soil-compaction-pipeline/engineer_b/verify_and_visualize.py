import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def verify_and_visualize(payload_path: str, master_df_path: str, output_dir: str):
    if not os.path.exists(payload_path):
        print(f"File not found: {payload_path}")
        return
        
    with open(payload_path, 'r') as f:
        payload = json.load(f)
        
    # Read payload for sanity checks and counts
    df_payload = pd.DataFrame(payload)
    print(f"Loaded {len(df_payload)} payload records.")
    
    # Read full parquet for heatmap plotting (includes max_subsoil_stress_mpa etc)
    df = pd.read_parquet(master_df_path)
    print(f"Loaded {len(df)} master_df records for plotting.")
    
    # Sanity Checks
    print("\n--- Sanity Checks ---")
    
    invalid_actions = set(df['action'].unique()) - {"Targeted Deep Tillage", "Monitor - Not Economically Viable", "None", "INVALID_DATA"}
    if invalid_actions:
        print(f"ERROR: Found invalid actions: {invalid_actions}")
    else:
        print("PASS: All actions are within valid enum.")
        
    if (df['pred_ripper_depth_cm'] < 0).any() or (df['pred_ripper_depth_cm'] > 60).any():
        print("ERROR: pred_ripper_depth_cm bounds violated [0, 60]")
    else:
        print("PASS: pred_ripper_depth_cm in bounds [0, 60] or NaN")
        
    print(f"\nAction counts:\n{df['action'].value_counts()}")
    
    # Visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Action distribution
    plt.figure(figsize=(8, 5))
    df['action'].value_counts().plot(kind='bar', color=['#94A3B8', '#16A34A', '#D97706', '#DC2626'])
    plt.title("Distribution of Recommended Actions")
    plt.ylabel("Pixel Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_actions.png"))
    plt.close()

    # 2. Depth correlation with ROI
    valid_df = df[df['action'] != 'INVALID_DATA'].copy()
    if not valid_df.empty:
        plt.figure(figsize=(8, 5))
        plt.scatter(valid_df['roi'], valid_df['pred_ripper_depth_cm'], alpha=0.6, c='blue')
        plt.axvline(x=1.2, color='r', linestyle='--', label='ROI Trigger (1.2)')
        plt.title("ROI vs Predicted Ripper Depth")
        plt.xlabel("Economic ROI")
        plt.ylabel("Predicted Ripper Depth (cm)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_roi_depth.png"))
        plt.close()

    # Determine Grid Size for Heatmaps
    n_pixels = len(df)
    # Assuming it's a perfect square based on generate_synthetic_data script
    grid_size = int(np.sqrt(n_pixels))
    
    if grid_size * grid_size == n_pixels:
        print(f"\nGenerating heatmaps for {grid_size}x{grid_size} farm...")
        # Columns requested by user
        cols_to_plot = [
            "max_subsoil_stress_mpa", 
            "depth_of_max_stress_cm", 
            "pred_ripper_depth_cm", 
            "mapie_lower_bound", 
            "mapie_upper_bound", 
            "roi"
        ]
        
        for col in cols_to_plot:
            plt.figure(figsize=(10, 8))
            # Reshape values into 2D grid. We replace None/NaN with 0 for plotting
            data_grid = df[col].fillna(0).values.reshape(grid_size, grid_size)
            
            sns.heatmap(data_grid, cmap='viridis', annot=False)
            plt.title(f"Heatmap: {col}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"heatmap_{col}.png"))
            plt.close()

        # Custom Action Heatmap!
        action_map = {
            "Targeted Deep Tillage": 3, # Red-ish
            "Monitor - Not Economically Viable": 2, # Orange-ish
            "None": 1, # Green-ish
            "INVALID_DATA": 0 # Grey
        }
        
        action_grid = df['action'].map(action_map).fillna(0).values.reshape(grid_size, grid_size)
        plt.figure(figsize=(10, 8))
        cmap = sns.color_palette(["#94A3B8", "#16A34A", "#D97706", "#DC2626"]) # grey, green, amber, red
        sns.heatmap(action_grid, cmap=cmap, annot=False, cbar=False)
        plt.title("Heatmap: Prescribed Action")
        plt.axis('off')
        
        # Custom legend for actions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#DC2626', label='Targeted Deep Tillage'),
            Patch(facecolor='#D97706', label='Monitor'),
            Patch(facecolor='#16A34A', label='None'),
            Patch(facecolor='#94A3B8', label='INVALID_DATA')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_action.png"))
        plt.close()
        
    print(f"Saved plots to {output_dir}")

if __name__ == "__main__":
    base_dir = r"c:\Users\foodg\OneDrive\Documents\Kalshi_projects\Agtech\soil-compaction-pipeline"
    payload_file = os.path.join(base_dir, "pipeline_data", "final_payload.json")
    master_df_file = os.path.join(base_dir, "pipeline_data", "master_df.parquet")
    out_dir = os.path.join(base_dir, "engineer_b", "visualizations")
    verify_and_visualize(payload_file, master_df_file, out_dir)
