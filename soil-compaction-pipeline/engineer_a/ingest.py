import asyncio
import json
import logging
import os
from datetime import datetime
import pandas as pd
import geopandas as gpd

from engineer_a.models import FieldBoundary, PipelineError
from engineer_a.api_cdse_ndvi import fetch_cdse_ndvi
from engineer_a.api_soilgrids_wcs import fetch_soilgrids_wcs
from engineer_a.api_asabe_telemetry import fetch_tractor_ops
from engineer_a.validate import validate_dataframe

logger = logging.getLogger(__name__)

def align_and_reproject(ndvi_gdf: gpd.GeoDataFrame, soil_gdf: gpd.GeoDataFrame, ops_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Stage 2 Spatial Alignment:
    Transforms all incoming EPSG:4326 GeoDataFrames to EPSG:3857 for precise meter-based
    Cartesian distances. Executes topological point-in-polygon and distance-nearest joins.
    """
    logger.info("Projecting DataFrames to EPSG:3857 for spatial alignment...")
    
    # Assert input CRS
    for name, gdf in [("NDVI", ndvi_gdf), ("SoilGrids", soil_gdf), ("ASABE", ops_gdf)]:
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            logger.warning(f"{name} missing CRS or not EPSG:4326. Auto-fixing.")
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
            
    # Project to Web Mercator (meters)
    master_m = ndvi_gdf.to_crs("EPSG:3857")
    soil_m = soil_gdf.to_crs("EPSG:3857")
    ops_m = ops_gdf.to_crs("EPSG:3857")
    
    logger.info(f"Alignment Starting. Base resolution pixels: {len(master_m)}")
    
    # 1. Join Soil (Point in Polygon if SoilGrids returns polys, or Nearest if Points)
    # The WCS returns 250m points in our implementation, so we do a Nearest join.
    logger.info("Joining SoilGrids (WCS) data...")
    if not soil_m.empty:
        # Distance nearest to get the 250m WCS pixel info to the 10m Sentinel pixel
        master_m = gpd.sjoin_nearest(master_m, soil_m, how="left", rsuffix="soil")
    else:
        master_m["clay_pct"] = pd.NA
        master_m["bulk_density"] = pd.NA
        
    master_m = master_m.drop(columns=["index_soil"], errors="ignore")

    # 2. Join Tractor Telemetry
    # We only want passes within 5 meters of the Sentinel-2 10x10m pixel centroid
    logger.info("Joining ASABE Tractor Telemetry (Max Distance: 5m)...")
    if not ops_m.empty:
        master_m = gpd.sjoin_nearest(master_m, ops_m, how="left", max_distance=5.0, rsuffix="ops")
    else:
        master_m["equipment_weight_kg"] = pd.NA
        master_m["tire_width_m"] = pd.NA
        
    master_m = master_m.drop(columns=["index_ops"], errors="ignore")
    
    # 3. Handle Duplicate Joins
    # sjoin_nearest can duplicate left points if there are equidistant right points.
    # Group by geometry to keep the grid pure
    original_len = len(master_m)
    master_m = master_m[~master_m.geometry.duplicated(keep="first")]
    if len(master_m) < original_len:
         logger.info(f"Dropped {original_len - len(master_m)} duplicated spatial joins.")

    # 4. Project back to WGS-84 (EPSG:4326) and flatten coords
    logger.info("Projecting unified DataFrame back to EPSG:4326...")
    master_gps = master_m.to_crs("EPSG:4326")
    
    master_gps["lat"] = master_gps.geometry.y
    master_gps["lon"] = master_gps.geometry.x
    
    # Drop the heavy Shapely objects before returning the Pandas DataFrame
    master_df = pd.DataFrame(master_gps.drop(columns=["geometry"]))
    
    # Assign standard pixel_id
    master_df["pixel_id"] = [f"px_{i:04d}" for i in range(len(master_df))]
    
    columns_order = [
        "pixel_id", "lat", "lon", "ndvi", "clay_pct", "bulk_density", 
        "equipment_weight_kg", "tire_width_m"
    ]
    
    # Ensure all required columns exist, fill missing with NaN
    for col in columns_order:
        if col not in master_df.columns:
            master_df[col] = pd.NA
            
    return master_df[columns_order]

async def ingest_and_align(geojson_polygon: dict, config_path: str = "config.yaml", out_dir: str = "pipeline_data") -> pd.DataFrame:
    """
    Stage 1 & 2 Orchestrator.
    Executes CDSE, SoilGrids, and ASABE data generation concurrently,
    then aligns them into a single validation-checked Parquet artifact.
    """
    start_time = datetime.utcnow()
    os.makedirs(out_dir, exist_ok=True)
    
    logger.info("Initializing Stage 1 Data Gathering...")
    boundary = FieldBoundary.from_geojson(geojson_polygon)
    
    # Concurrent gathering with asyncio
    sources = ["API A: CDSE (NDVI)", "API B: SoilGrids (WCS)", "API C: ASABE (Telemetry)"]
    results = await asyncio.gather(
        fetch_cdse_ndvi(boundary),
        fetch_soilgrids_wcs(boundary),
        fetch_tractor_ops(boundary, config_path),
        return_exceptions=True
    )
    
    manifest_sources = {}
    valid_results = []
    
    # Process return exceptions
    for source_name, p_result in zip(sources, results):
        if isinstance(p_result, Exception):
            logger.error(f"{source_name} orchestrator captured exception: {p_result}")
            manifest_sources[source_name] = f"FAILED: {str(p_result)}"
            # If our individual APIs failed to provide fallbacks, inject empty frame
            valid_results.append(gpd.GeoDataFrame(crs="EPSG:4326"))
        else:
            manifest_sources[source_name] = "SUCCESS"
            valid_results.append(p_result)
            
    ndvi_gdf, soil_gdf, ops_gdf = valid_results
    
    # Standard Alignment 
    master_df = align_and_reproject(ndvi_gdf, soil_gdf, ops_gdf)
    
    # Validation Gauntlet
    logger.info("Running Validation Gauntlet...")
    master_df = validate_dataframe(master_df)
    
    # Terminal State Check
    valid_pixels = master_df["data_valid"].sum()
    if valid_pixels == 0:
        logger.error("Zero valid pixels after Validation Gauntlet.")
        raise PipelineError("Pipeline Halted: Zero actionable pixels generated.")
        
    # Serialize Parquet Contract
    parquet_path = os.path.join(out_dir, "master_df.parquet")
    logger.info(f"Serializing {valid_pixels} valid pixels to {parquet_path}...")
    master_df.to_parquet(parquet_path, engine="pyarrow", index=False)
    
    # Serialize Manifest
    manifest = {
        "timestamp_utc": start_time.isoformat(),
        "total_pixels_generated": len(master_df),
        "valid_pixels": int(valid_pixels),
        "validation_pass_rate_pct": float(valid_pixels / len(master_df)) * 100,
        "api_health": manifest_sources
    }
    
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    return master_df
