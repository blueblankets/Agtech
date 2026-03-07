import pandas as pd
import logging
from engineer_b.constants import GLOBAL_BOUNDS

logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the validation gauntlet defined by the data contracts.
    Out-of-bound values trigger data_valid=False for that specific pixel,
    but the pipeline continues.
    """
    
    # Initialize all as valid
    df["data_valid"] = True
    df["invalid_fields"] = ""
    
    bounds_map = {
        "ndvi": GLOBAL_BOUNDS["ndvi"],
        "clay_pct": GLOBAL_BOUNDS["clay_pct"],
        "bulk_density": GLOBAL_BOUNDS["bulk_density"],
        "equipment_weight_kg": GLOBAL_BOUNDS["equipment_weight_kg"],
        "tire_width_m": GLOBAL_BOUNDS["tire_width_m"]
    }

    # First, handle the scaling edge case purely for SoilGrids (e.g. unscaled integer 1450 instead of 1.45)
    if "bulk_density" in df.columns:
        unscaled_mask = df["bulk_density"] > 10.0
        if unscaled_mask.any():
            logger.warning(f"Found {unscaled_mask.sum()} unscaled bulk_density values > 10. Auto-applying ÷1000 transform.")
            df.loc[unscaled_mask, "bulk_density"] = df.loc[unscaled_mask, "bulk_density"] / 1000.0

    # Apply strict physical limits
    for col, (lo, hi) in bounds_map.items():
        if col not in df.columns: continue
        
        # We only invalidate if it's NOT NaN AND it's out of bounds.
        # equipment specs being NaN is valid (meaning no tractor pass in that area)
        mask_populated = df[col].notna()
        mask_out_of_bounds = mask_populated & ~df[col].between(lo, hi)
        
        if mask_out_of_bounds.any():
            fail_count = mask_out_of_bounds.sum()
            logger.warning(f"{fail_count} pixels failed {col} validation (range [{lo},{hi}]).")
            
            # Flag rows
            df.loc[mask_out_of_bounds, "data_valid"] = False
            # Append failed column to comma separated string
            current_str = df.loc[mask_out_of_bounds, "invalid_fields"]
            df.loc[mask_out_of_bounds, "invalid_fields"] = current_str + col + ","
            
    # Clean up trailing commas
    df["invalid_fields"] = df["invalid_fields"].str.rstrip(",")
    
    valid_count = df["data_valid"].sum()
    logger.info(f"Validation complete: {valid_count}/{len(df)} pixels passed.")
    
    return df
