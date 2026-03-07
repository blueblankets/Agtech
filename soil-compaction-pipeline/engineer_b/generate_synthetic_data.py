import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

np.random.seed(42)

def generate_synthetic_data(output_dir: str):
    num_rows, num_cols = 50, 50
    total_pixels = num_rows * num_cols
    
    # Base coords (~Iowa farm)
    base_lat = 41.5
    base_lon = -93.5
    
    # 10m in degrees is roughly 0.00009
    lat_step = 0.00009
    lon_step = 0.00009
    
    lats = []
    lons = []
    for r in range(num_rows):
        for c in range(num_cols):
            lats.append(base_lat + r * lat_step)
            lons.append(base_lon + c * lon_step)
            
    # Synthetic values
    ndvi = np.random.uniform(0.4, 0.9, total_pixels)
    clay_pct = np.random.uniform(20.0, 60.0, total_pixels)
    bulk_density = np.random.uniform(1.3, 1.8, total_pixels)
    
    # Equipment weight: make them heavier to trigger higher stresses/ROI
    equipment_weight = np.random.uniform(15000.0, 45000.0, total_pixels)
    weight_nans = np.random.choice([True, False], total_pixels, p=[0.2, 0.8])
    equipment_weight[weight_nans] = np.nan
    
    # Tire width
    tire_width = np.random.uniform(0.5, 1.2, total_pixels)
    tire_width[weight_nans] = np.nan
    
    df = pd.DataFrame({
        "pixel_id": [f"px_{i:04d}" for i in range(total_pixels)],
        "lat": lats,
        "lon": lons,
        "ndvi": ndvi,
        "clay_pct": clay_pct,
        "bulk_density": bulk_density,
        "equipment_weight_kg": equipment_weight,
        "tire_width_m": tire_width,
        "data_valid": True,
        "invalid_fields": ""
    })
    
    # Force one pixel to be explicitly invalid to test pipeline
    df.loc[0, "equipment_weight_kg"] = 50000.0  # Out of bounds test
    df.loc[0, "data_valid"] = False
    df.loc[0, "invalid_fields"] = "equipment_weight_kg"
    
    # Force one to have very low NDVI to test "Monitor" and "Targeted Deep Tillage" action logic
    df.loc[1, "ndvi"] = 0.60
    df.loc[1, "equipment_weight_kg"] = 35000.0
    df.loc[1, "bulk_density"] = 1.7
    
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "master_df.parquet")
    df.to_parquet(parquet_path)
    print(f"Saved {len(df)} pixels to {parquet_path}")
    
    manifest = {
        "pixel_count": total_pixels,
        "valid_count": int(df["data_valid"].sum()),
        "api_sources_used": ["mock_synthetic"],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

if __name__ == "__main__":
    generate_synthetic_data(r"c:\Users\foodg\OneDrive\Documents\Kalshi_projects\Agtech\soil-compaction-pipeline\pipeline_data")
