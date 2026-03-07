"""
API A — CDSE openEO: Sentinel-2 L2A NDVI Ingestion.

Fetches cloud-masked NDVI median composite from the Copernicus Data Space
Ecosystem. Falls back to local synthetic generation ONLY if the real API
is unreachable, and logs transparently when this happens.
"""
import openeo
import geopandas as gpd
import numpy as np
import asyncio
import logging
import io
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point
from engineer_a.models import FieldBoundary, PipelineError
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Module-level flag so manifest can report honestly
FALLBACK_USED = False


def get_auth_connection():
    """Authenticate with CDSE openEO via OIDC browser flow."""
    logger.info("Connecting to CDSE openEO...")
    conn = openeo.connect("openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()
    logger.info("CDSE OIDC authentication successful")
    return conn


def _build_synthetic_ndvi(boundary: FieldBoundary) -> gpd.GeoDataFrame:
    """
    LOCAL synthetic NDVI generation — used ONLY when CDSE is unreachable.

    Generates a spatially-coherent NDVI gradient using realistic agricultural
    patterns (field-edge stress from headland compaction, drainage effects).

    *** THIS IS NOT REAL SATELLITE DATA — manifest will say SYNTHETIC ***
    """
    logger.warning("=" * 60)
    logger.warning("CDSE API UNAVAILABLE — GENERATING SYNTHETIC NDVI")
    logger.warning("This is NOT real satellite data.")
    logger.warning("=" * 60)

    global FALLBACK_USED
    FALLBACK_USED = True

    minx, miny, maxx, maxy = boundary.get_bounds()
    step = 0.00009  # ~10m pixel spacing

    lons = np.arange(minx, maxx, step)
    lats = np.arange(miny, maxy, step)

    records = []
    for lon in lons:
        for lat in lats:
            pt = Point(lon, lat)
            if boundary.geometry.contains(pt):
                # Spatially coherent gradient simulating:
                #   - Higher NDVI in field center (healthy crop)
                #   - Lower NDVI at edges (compaction from headland turns)
                #   - Drainage-driven dip in lower-left quadrant
                norm_x = (lon - minx) / (maxx - minx + 1e-9)
                norm_y = (lat - miny) / (maxy - miny + 1e-9)

                # Distance from center (0=center, 1=corner)
                dist_center = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2) / 0.707

                # Base NDVI with edge stress (headlands get more traffic)
                base_ndvi = 0.75 - 0.20 * dist_center

                # Drainage dip in lower-left (simulates wet spot / poor drainage)
                drainage = -0.10 * np.exp(-((norm_x - 0.25)**2 + (norm_y - 0.25)**2) / 0.02)

                # Small-scale field variability
                noise = np.random.normal(0, 0.03)

                ndvi = float(np.clip(base_ndvi + drainage + noise, 0.15, 0.92))
                records.append({"ndvi": np.float32(ndvi), "geometry": pt})

    if not records:
        records.append({"ndvi": np.float32(0.65), "geometry": boundary.geometry.centroid})

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    logger.info("Synthetic NDVI: %d pixels generated", len(gdf))
    return gdf


async def fetch_cdse_ndvi(
    boundary: FieldBoundary,
    date_range: tuple = None,
    cloud_threshold: int = 20,
) -> gpd.GeoDataFrame:
    """
    Fetch NDVI from CDSE openEO. Attempts the REAL API first.
    Falls back to synthetic generation ONLY on actual failure.
    """
    try:
        gdf = await asyncio.to_thread(
            _fetch_cdse_ndvi_sync, boundary, date_range, cloud_threshold
        )
        logger.info("CDSE returned LIVE data: %d pixels", len(gdf))
        return gdf
    except Exception as e:
        logger.error("CDSE NDVI fetch FAILED: %s", e)
        logger.error("Falling back to synthetic NDVI generation")
        return _build_synthetic_ndvi(boundary)


def _fetch_cdse_ndvi_sync(
    boundary: FieldBoundary,
    date_range: tuple,
    cloud_threshold: int,
) -> gpd.GeoDataFrame:
    """Synchronous CDSE fetch — runs in a thread."""
    conn = get_auth_connection()

    if date_range is None:
        end = datetime.utcnow()
        start = end - timedelta(days=14)
        date_range = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    bbox = boundary.get_bounds()
    spatial_extent = {
        "west": bbox[0], "south": bbox[1],
        "east": bbox[2], "north": bbox[3],
    }

    logger.info("Submitting openEO graph: NDVI %s, dates %s", spatial_extent, date_range)

    # Build the processing graph
    cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=list(date_range),
        bands=["B04", "B08", "SCL"],
    )

    # SCL cloud mask: keep vegetation (4) and bare soil (5)
    scl = cube.band("SCL")
    mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11)
    cube = cube.mask(mask)

    # NDVI = (B08 - B04) / (B08 + B04)
    b08 = cube.band("B08")
    b04 = cube.band("B04")
    ndvi = (b08 - b04) / (b08 + b04)

    # Temporal median
    ndvi_median = ndvi.reduce_dimension(dimension="t", reducer="median")

    # Download as GeoTIFF into memory
    result_bytes = ndvi_median.download(format="GTiff")
    logger.info("CDSE download complete: %d bytes", len(result_bytes))

    # Parse GeoTIFF -> GeoDataFrame
    gdf = _geotiff_to_geodataframe(result_bytes)

    if len(gdf) == 0:
        raise PipelineError("CDSE returned 0 valid NDVI pixels after cloud masking")

    logger.info("CDSE NDVI: %d LIVE pixels retrieved", len(gdf))
    return gdf


def _geotiff_to_geodataframe(tiff_bytes: bytes) -> gpd.GeoDataFrame:
    """Convert GeoTIFF bytes to a GeoDataFrame of point geometries."""
    with rasterio.open(io.BytesIO(tiff_bytes)) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        valid_mask = np.isfinite(data)
        if nodata is not None:
            valid_mask &= (data != nodata)

        rows_idx, cols_idx = np.where(valid_mask)
        if len(rows_idx) == 0:
            return gpd.GeoDataFrame(
                columns=["geometry", "ndvi"], geometry="geometry", crs="EPSG:4326"
            )

        xs, ys = xy(transform, rows_idx, cols_idx)
        values = data[rows_idx, cols_idx]
        points = [Point(x, y) for x, y in zip(xs, ys)]

        gdf = gpd.GeoDataFrame(
            {"ndvi": values.astype(np.float32)}, geometry=points, crs=crs
        )
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
