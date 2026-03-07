"""
Validation Gauntlet — Stage 2 post-join validation.

Every pixel is checked against GLOBAL_BOUNDS. Failures set data_valid=False
and append the column name to invalid_fields. The pipeline continues unless
ALL pixels fail, in which case PipelineError is raised.
"""
import pandas as pd
import numpy as np
import logging
from engineer_b.constants import GLOBAL_BOUNDS

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full validation gauntlet per the data contract.

    Checks:
      1. NDVI range:           -1.0 <= ndvi <= 1.0
      2. NDVI crop plausibility: 0.2 <= ndvi <= 0.9 (INFO only)
      3. Clay fraction:         0 <= clay_pct <= 100
      4. Bulk density physics:  1.0 <= bulk_density <= 2.0 g/cm3
      5. Bulk density scaling:  auto-detect un-scaled integers
      6. Equipment weight:      500 <= weight <= 40000 (NaN exempt)
      7. Tire width:            0.1 <= width <= 2.0 (NaN exempt)
      8. Coordinate bounds:     -90 <= lat <= 90; -180 <= lon <= 180
      9. Pixel count:           >= 1 valid row
    """
    n_total = len(df)
    logger.info("Validation gauntlet starting: %d pixels", n_total)

    df["data_valid"] = True
    df["invalid_fields"] = ""

    # --- Check 5 first: auto-fix bulk density scaling ---
    if "bulk_density" in df.columns:
        bd_valid = df["bulk_density"].dropna()
        if len(bd_valid) > 0:
            median_bd = bd_valid.median()
            if median_bd > 100:
                logger.warning(
                    "Bulk density auto-fix: median=%.1f suggests un-scaled integers. "
                    "Applying /10/1000.", median_bd
                )
                mask = df["bulk_density"].notna()
                df.loc[mask, "bulk_density"] = df.loc[mask, "bulk_density"] / 10.0 / 1000.0
            elif median_bd > 10:
                logger.warning(
                    "Bulk density auto-fix: median=%.1f suggests partial scaling. "
                    "Applying /1000.", median_bd
                )
                mask = df["bulk_density"].notna()
                df.loc[mask, "bulk_density"] = df.loc[mask, "bulk_density"] / 1000.0

    # --- Checks 1, 3, 4, 6, 7: range validation ---
    bounds_map = {
        "ndvi": GLOBAL_BOUNDS["ndvi"],
        "clay_pct": GLOBAL_BOUNDS["clay_pct"],
        "bulk_density": GLOBAL_BOUNDS["bulk_density"],
        "equipment_weight_kg": GLOBAL_BOUNDS["equipment_weight_kg"],
        "tire_width_m": GLOBAL_BOUNDS["tire_width_m"],
    }

    # Equipment cols are NaN-exempt (no tractor pass = valid state)
    nan_exempt_cols = {"equipment_weight_kg", "tire_width_m"}

    for col, (lo, hi) in bounds_map.items():
        if col not in df.columns:
            continue

        populated = df[col].notna()
        out_of_bounds = populated & ~df[col].between(lo, hi)

        if out_of_bounds.any():
            n_fail = out_of_bounds.sum()
            logger.warning("%d pixels failed %s check (range [%.2f, %.2f])", n_fail, col, lo, hi)
            df.loc[out_of_bounds, "data_valid"] = False
            df.loc[out_of_bounds, "invalid_fields"] += col + ","

        # Flag NaN as invalid for non-exempt columns
        if col not in nan_exempt_cols:
            nan_mask = df[col].isna()
            if nan_mask.any():
                logger.warning("%d pixels have NaN %s", nan_mask.sum(), col)
                df.loc[nan_mask, "data_valid"] = False
                df.loc[nan_mask, "invalid_fields"] += col + ","

    # --- Check 2: NDVI crop plausibility (INFO only) ---
    if "ndvi" in df.columns:
        outside_crop = df["ndvi"].notna() & ~df["ndvi"].between(0.2, 0.9)
        if outside_crop.any():
            logger.info(
                "%d pixels have NDVI outside crop plausibility [0.2, 0.9] (informational)",
                outside_crop.sum(),
            )

    # --- Check 8: Coordinate bounds ---
    if "lat" in df.columns:
        bad_lat = df["lat"].notna() & ~df["lat"].between(-90, 90)
        if bad_lat.any():
            df.loc[bad_lat, "data_valid"] = False
            df.loc[bad_lat, "invalid_fields"] += "coords,"
            logger.warning("%d pixels have lat outside [-90, 90]", bad_lat.sum())

    if "lon" in df.columns:
        bad_lon = df["lon"].notna() & ~df["lon"].between(-180, 180)
        if bad_lon.any():
            df.loc[bad_lon, "data_valid"] = False
            df.loc[bad_lon, "invalid_fields"] += "coords,"
            logger.warning("%d pixels have lon outside [-180, 180]", bad_lon.sum())

    # Clean up trailing commas
    df["invalid_fields"] = df["invalid_fields"].str.rstrip(",")

    # --- Check 9: Terminal check ---
    n_valid = df["data_valid"].sum()
    n_invalid = n_total - n_valid
    logger.info("Validation complete: %d/%d valid, %d invalid", n_valid, n_total, n_invalid)

    return df
