"""
Engineer A — Top-level Orchestrator (Stages 1 & 2).

Receives a GeoJSON polygon, concurrently fetches NDVI (CDSE), soil (SoilGrids),
and tractor telemetry (ASABE), performs spatial alignment via geopandas joins,
runs the validation gauntlet, and emits master_df.parquet + manifest.json.
"""
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


def align_and_reproject(
    ndvi_gdf: gpd.GeoDataFrame,
    soil_gdf: gpd.GeoDataFrame,
    ops_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Stage 2 Spatial Alignment:
    Transforms all EPSG:4326 GeoDataFrames to EPSG:3857 for metre-based
    Cartesian joins, then reprojects back to EPSG:4326.
    """
    logger.info("Projecting DataFrames to EPSG:3857 for spatial alignment...")

    # Assert input CRS
    for name, gdf in [("NDVI", ndvi_gdf), ("SoilGrids", soil_gdf), ("ASABE", ops_gdf)]:
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            logger.warning("%s CRS missing or not EPSG:4326. Auto-fixing.", name)
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)

    # Handle empty NDVI — cannot proceed without base grid
    if ndvi_gdf.empty:
        raise PipelineError("NDVI GeoDataFrame is empty — cannot build pixel grid")

    # Project to Web Mercator (metres)
    master_m = ndvi_gdf.to_crs("EPSG:3857")
    soil_m = soil_gdf.to_crs("EPSG:3857") if not soil_gdf.empty else soil_gdf
    ops_m = ops_gdf.to_crs("EPSG:3857") if not ops_gdf.empty else ops_gdf

    logger.info("Alignment: %d NDVI pixels, %d soil cells, %d tractor points",
                len(master_m), len(soil_m), len(ops_m))

    # 1. Join Soil (nearest join — 250m WCS points to 10m NDVI pixels)
    if not soil_m.empty:
        logger.info("Joining SoilGrids data (nearest)...")
        master_m = gpd.sjoin_nearest(master_m, soil_m, how="left", rsuffix="soil")
        master_m = master_m.drop(columns=["index_soil"], errors="ignore")
    else:
        logger.warning("Soil GDF empty — clay_pct and bulk_density will be NaN")
        master_m["clay_pct"] = pd.NA
        master_m["bulk_density"] = pd.NA

    # 2. Join Tractor Telemetry (nearest — no distance limit because
    #    the tractor covers the entire field; every pixel is affected by
    #    the equipment that traversed nearest to it)
    if not ops_m.empty:
        logger.info("Joining ASABE telemetry (nearest, field-wide)...")
        master_m = gpd.sjoin_nearest(master_m, ops_m, how="left", rsuffix="ops")
        master_m = master_m.drop(columns=["index_ops"], errors="ignore")
    else:
        logger.warning("Tractor GDF empty — equipment cols will be NaN (valid state)")
        master_m["equipment_weight_kg"] = pd.NA
        master_m["tire_width_m"] = pd.NA

    # 3. Deduplicate spatial joins
    original_len = len(master_m)
    master_m = master_m[~master_m.geometry.duplicated(keep="first")]
    if len(master_m) < original_len:
        logger.info("Dropped %d duplicated spatial joins", original_len - len(master_m))

    # 4. Project back to WGS-84
    master_gps = master_m.to_crs("EPSG:4326")
    master_gps["lat"] = master_gps.geometry.y
    master_gps["lon"] = master_gps.geometry.x

    master_df = pd.DataFrame(master_gps.drop(columns=["geometry"]))
    master_df["pixel_id"] = [f"px_{i:04d}" for i in range(len(master_df))]

    columns_order = [
        "pixel_id", "lat", "lon", "ndvi", "clay_pct", "bulk_density",
        "equipment_weight_kg", "tire_width_m",
    ]
    for col in columns_order:
        if col not in master_df.columns:
            master_df[col] = pd.NA

    return master_df[columns_order]


async def ingest_and_align(
    geojson_polygon: dict,
    config_path: str = "config.yaml",
    out_dir: str = "pipeline_data",
) -> pd.DataFrame:
    """
    Stage 1 & 2 Orchestrator.
    Concurrent data ingestion → spatial alignment → validation → Parquet emit.
    """
    start_time = datetime.utcnow()
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Initializing Stage 1 Data Gathering...")
    boundary = FieldBoundary.from_geojson(geojson_polygon)

    # Concurrent gathering
    sources = ["API A: CDSE (NDVI)", "API B: SoilGrids (WCS)", "API C: ASABE (Telemetry)"]
    results = await asyncio.gather(
        fetch_cdse_ndvi(boundary),
        fetch_soilgrids_wcs(boundary),
        fetch_tractor_ops(boundary, config_path),
        return_exceptions=True,
    )

    manifest_sources = {}
    valid_results = []

    for source_name, p_result in zip(sources, results):
        if isinstance(p_result, Exception):
            logger.error("%s orchestrator exception: %s", source_name, p_result)
            manifest_sources[source_name] = f"FAILED: {str(p_result)}"
            valid_results.append(gpd.GeoDataFrame(crs="EPSG:4326"))
        else:
            valid_results.append(p_result)
            # Check if fallback was used (transparent reporting)
            if source_name == "API A: CDSE (NDVI)":
                from engineer_a.api_cdse_ndvi import FALLBACK_USED as cdse_fb
                manifest_sources[source_name] = "SYNTHETIC (CDSE unreachable)" if cdse_fb else "LIVE"
            elif source_name == "API B: SoilGrids (WCS)":
                from engineer_a.api_soilgrids_wcs import FALLBACK_USED as soil_fb
                manifest_sources[source_name] = "SYNTHETIC (WCS unreachable)" if soil_fb else "LIVE"
            else:
                manifest_sources[source_name] = f"SIMULATED ({len(p_result)} points)"

    ndvi_gdf, soil_gdf, ops_gdf = valid_results

    logger.info("NDVI: %d features, Soil: %d features, Tractor: %d features",
                len(ndvi_gdf), len(soil_gdf), len(ops_gdf))

    # Spatial alignment
    master_df = align_and_reproject(ndvi_gdf, soil_gdf, ops_gdf)

    # Validation gauntlet
    logger.info("Running Validation Gauntlet...")
    master_df = validate_dataframe(master_df)

    # Terminal check
    valid_pixels = master_df["data_valid"].sum()
    if valid_pixels == 0:
        raise PipelineError("Pipeline Halted: Zero actionable pixels after validation.")

    # Emit Parquet
    parquet_path = os.path.join(out_dir, "master_df.parquet")
    logger.info("Serializing %d valid pixels to %s", valid_pixels, parquet_path)
    master_df.to_parquet(parquet_path, engine="pyarrow", index=False)

    # Emit transparent manifest
    manifest = {
        "timestamp_utc": start_time.isoformat(),
        "total_pixels_generated": len(master_df),
        "valid_pixels": int(valid_pixels),
        "validation_pass_rate_pct": float(valid_pixels / len(master_df)) * 100,
        "api_health": manifest_sources,
        "transparency_note": "LIVE = real API data. SYNTHETIC = locally generated fallback. SIMULATED = ASABE algorithmic generator (by design).",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    logger.info("Manifest written. API health: %s", manifest_sources)
    return master_df
