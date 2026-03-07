import pandas as pd
import numpy as np
import json
import os
import math
import logging
import sys

# Add parent directory to path to allow importing engineer_b
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engineer_b.physics import sohne_stress
from engineer_b.ml_inference import run_ml_inference
from engineer_b.economic_filter import determine_action
from engineer_b.constants import calculate_roi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_model_pipeline(master_df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    # Initialize output columns
    master_df["max_subsoil_stress_mpa"] = np.nan
    master_df["depth_of_max_stress_cm"] = np.nan
    master_df["pred_ripper_depth_cm"] = np.nan
    master_df["mapie_lower_bound"] = np.nan
    master_df["mapie_upper_bound"] = np.nan
    master_df["roi"] = 0.0
    master_df["action"] = "None"
    
    DEPTHS_CM = [10, 20, 30]
    mapie_path = os.path.join(model_dir, "mapie_model.pkl")
    if not os.path.exists(mapie_path):
        raise FileNotFoundError(f"Model files missing at {mapie_path}")

    for idx, row in master_df.iterrows():
        if not row["data_valid"]:
            master_df.at[idx, "action"] = "INVALID_DATA"
            continue

        # Stage 3: Physics
        max_stress = float('nan')
        depth_of_max = float('nan')
        if pd.notna(row["equipment_weight_kg"]):
            stresses = [
                sohne_stress(d, row["equipment_weight_kg"], row["tire_width_m"], row["bulk_density"], row["clay_pct"])
                for d in DEPTHS_CM
            ]
            valid_stresses = [s for s in stresses if not math.isnan(s)]
            if valid_stresses:
                max_stress = max(valid_stresses)
                # Cap at 5.0 MPa constraint
                if max_stress > 5.0:
                    logger.warning(f"Pixel {row['pixel_id']}: stress {max_stress} capped to 5.0")
                    max_stress = 5.0
                depth_of_max = DEPTHS_CM[stresses.index(max_stress) if max_stress in stresses else 0]
        else:
            master_df.at[idx, "action"] = "None"
            continue
        
        master_df.at[idx, "max_subsoil_stress_mpa"] = max_stress
        master_df.at[idx, "depth_of_max_stress_cm"] = depth_of_max
        
        # Stage 4: ML
        features = [row["ndvi"], row["clay_pct"], row["bulk_density"], max_stress]
        if any(pd.isna(f) for f in features):
            master_df.at[idx, "action"] = "INVALID_DATA"
            logger.warning(f"Pixel {row['pixel_id']}: NaN in ML features, skipping.")
            continue
            
        depth, lo, hi = run_ml_inference(features, mapie_path)
        master_df.at[idx, "pred_ripper_depth_cm"] = depth
        master_df.at[idx, "mapie_lower_bound"] = lo
        master_df.at[idx, "mapie_upper_bound"] = hi
        
        # Stage 5: Economic filter
        roi = calculate_roi(max_stress)
        master_df.at[idx, "roi"] = roi
        master_df.at[idx, "action"] = determine_action(roi, hi, row["ndvi"])

    return master_df

def save_final_payload(master_df: pd.DataFrame, output_dir: str):
    records = []
    # final_payload.json: array of {pixel_id, lat, lon, action, pred_ripper_depth_cm, mapie_lower_bound, mapie_upper_bound, roi}
    cols = ["pixel_id", "lat", "lon", "action", "pred_ripper_depth_cm", "mapie_lower_bound", "mapie_upper_bound", "roi"]
    for _, row in master_df.iterrows():
        rec = {}
        for c in cols:
            val = row[c]
            # Replace NaN with None for JSON serialization
            if pd.isna(val):
                rec[c] = None
            else:
                rec[c] = val
        records.append(rec)
        
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "final_payload.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {out_path}")

def main():
    base_dir = r"c:\Users\foodg\OneDrive\Documents\Kalshi_projects\Agtech\soil-compaction-pipeline"
    pipeline_data_dir = os.path.join(base_dir, "pipeline_data")
    engineer_b_dir = os.path.join(base_dir, "engineer_b")
    
    parquet_path = os.path.join(pipeline_data_dir, "master_df.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    print(f"Loaded master_df with {len(df)} rows.")
    
    df_out = run_model_pipeline(df, engineer_b_dir)
    
    # Save back to parquet
    df_out.to_parquet(parquet_path)
    
    # Generate final JSON payload
    save_final_payload(df_out, pipeline_data_dir)
    print("Engineer B workflow complete.")

if __name__ == "__main__":
    main()
