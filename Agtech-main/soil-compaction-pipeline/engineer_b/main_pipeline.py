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
from engineer_b.ml_inference import load_mapie_model, run_ml_inference_batch
from engineer_b.economic_filter import determine_action
from engineer_b.constants import (
    calculate_roi, ROI_TRIGGER_THRESHOLD, NDVI_STRESS_THRESHOLD,
    TILLAGE_COST_PER_ACRE, COMPACTION_LOSS_PER_ACRE,
    STRESS_FULL_DAMAGE_MPA,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _vectorized_sohne_stress(z_cm, weight_kg, tire_width_m, bulk_density, clay_pct):
    """Vectorized Söhne (1953) vertical stress propagation."""
    z_m = z_cm / 100.0

    pressure_pa = 150000.0 + (clay_pct * 500.0)
    contact_area_m2 = (weight_kg * 9.81) / pressure_pa
    contact_area_m2 = np.maximum(contact_area_m2, 1e-10)  # avoid div by zero

    radius_m = np.sqrt(contact_area_m2 / np.pi)

    # Söhne concentration factor k (bulk_density dependent)
    k = np.where(bulk_density < 1.4, 4.0, np.where(bulk_density < 1.6, 5.0, 6.0))

    sigma_pa = (weight_kg * 9.81 / contact_area_m2) * (1 - (z_m / np.sqrt(z_m**2 + radius_m**2))**k)
    return sigma_pa / 1e6  # Pa -> MPa


def run_model_pipeline(master_df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    import time
    t0 = time.time()

    # Initialize output columns
    n = len(master_df)
    master_df["max_subsoil_stress_mpa"] = np.nan
    master_df["depth_of_max_stress_cm"] = np.nan
    master_df["pred_ripper_depth_cm"] = np.nan
    master_df["mapie_lower_bound"] = np.nan
    master_df["mapie_upper_bound"] = np.nan
    master_df["roi"] = 0.0
    master_df["action"] = "None"

    # ─── Pre-load model ONCE ───
    mapie_path = os.path.join(model_dir, "mapie_model.pkl")
    if not os.path.exists(mapie_path):
        raise FileNotFoundError(f"Model files missing at {mapie_path}")
    logger.info("Loading MAPIE model (once)...")
    mapie_model = load_mapie_model(mapie_path)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # ─── Stage 3: Vectorized Physics ───
    t1 = time.time()
    valid_mask = master_df["data_valid"].fillna(False).astype(bool)
    has_equip = valid_mask & master_df["equipment_weight_kg"].notna()

    master_df.loc[~valid_mask, "action"] = "INVALID_DATA"

    if has_equip.any():
        weight = master_df.loc[has_equip, "equipment_weight_kg"].values.astype(float)
        tire_w = master_df.loc[has_equip, "tire_width_m"].values.astype(float)
        bd = master_df.loc[has_equip, "bulk_density"].values.astype(float)
        clay = master_df.loc[has_equip, "clay_pct"].values.astype(float)

        DEPTHS_CM = [10.0, 20.0, 30.0]
        stress_all = np.column_stack([
            _vectorized_sohne_stress(d, weight, tire_w, bd, clay) for d in DEPTHS_CM
        ])
        # Cap at 5.0 MPa
        stress_all = np.clip(stress_all, 0, 5.0)

        max_stress = np.max(stress_all, axis=1)
        depth_of_max_idx = np.argmax(stress_all, axis=1)
        depth_of_max = np.array(DEPTHS_CM)[depth_of_max_idx]

        master_df.loc[has_equip, "max_subsoil_stress_mpa"] = max_stress
        master_df.loc[has_equip, "depth_of_max_stress_cm"] = depth_of_max
    logger.info("Physics complete in %.1fs", time.time() - t1)

    # ─── Stage 4: Batch ML Inference ───
    t2 = time.time()
    # Build ML feature mask: has equipment + all features non-null
    ml_cols = ["ndvi", "clay_pct", "bulk_density", "max_subsoil_stress_mpa"]
    ml_ready = has_equip.copy()
    for col in ml_cols:
        ml_ready = ml_ready & master_df[col].notna()

    if ml_ready.any():
        X = master_df.loc[ml_ready, ml_cols].values.astype(float)
        logger.info("Running batch ML inference on %d pixels...", len(X))

        depths, lo, hi = run_ml_inference_batch(X, mapie_model)

        master_df.loc[ml_ready, "pred_ripper_depth_cm"] = depths
        master_df.loc[ml_ready, "mapie_lower_bound"] = lo
        master_df.loc[ml_ready, "mapie_upper_bound"] = hi
    logger.info("ML inference complete in %.1fs", time.time() - t2)

    # ─── Stage 5: Vectorized Economic Filter ───
    t3 = time.time()
    if ml_ready.any():
        stress_vals = master_df.loc[ml_ready, "max_subsoil_stress_mpa"].values

        # Vectorized calculate_roi (linear from 0 to STRESS_FULL_DAMAGE_MPA)
        loss_fraction = np.clip(stress_vals / STRESS_FULL_DAMAGE_MPA, 0.0, 1.0)
        avoided_loss = loss_fraction * COMPACTION_LOSS_PER_ACRE
        roi_vals = avoided_loss / TILLAGE_COST_PER_ACRE
        master_df.loc[ml_ready, "roi"] = roi_vals

        ndvi_vals = master_df.loc[ml_ready, "ndvi"].values
        hi_vals = master_df.loc[ml_ready, "mapie_upper_bound"].values

        # Vectorized action determination
        actions = np.where(
            (hi_vals > 0) & (ndvi_vals < NDVI_STRESS_THRESHOLD),
            np.where(roi_vals > ROI_TRIGGER_THRESHOLD, "Targeted Deep Tillage", "Monitor - Not Economically Viable"),
            "None"
        )
        master_df.loc[ml_ready, "action"] = actions
    logger.info("Economic filter complete in %.1fs", time.time() - t3)

    total = time.time() - t0
    logger.info("═══ Engineer B pipeline complete: %d pixels in %.1fs (%.0f px/sec) ═══", n, total, n / total if total > 0 else 0)

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
