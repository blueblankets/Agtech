"""
API B — ISRIC SoilGrids v2.0 WCS: Soil Mechanics Ingestion.

Fetches clay fraction (%) and bulk density (g/cm3) from ISRIC SoilGrids
OGC Web Coverage Service at 250m resolution.

CRITICAL UNIT SCALING (from SoilGrids documentation):
  clay : raw unit = g/kg (mapped value)  -> divide by 10 to get %
  bdod : raw unit = cg/cm3              -> divide by 100 to get g/cm3

If the WCS is unreachable, falls back to synthetic generation and logs
TRANSPARENTLY that it's synthetic.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.io import MemoryFile
from owslib.wcs import WebCoverageService
import asyncio
import logging
import urllib.request
from engineer_a.models import FieldBoundary, PipelineError

logger = logging.getLogger(__name__)

# SoilGrids WCS endpoints (separate map files per property)
SOILGRIDS_WCS_TEMPLATE = "https://maps.isric.org/mapserv?map=/map/{property}.map"
COVERAGEID_CLAY = "clay_5-15cm_Q0.5"
COVERAGEID_BDOD = "bdod_5-15cm_Q0.5"
FORMAT = "GEOTIFF_INT16"

# Module-level flag for manifest transparency
FALLBACK_USED = False


def _build_synthetic_soil(boundary: FieldBoundary) -> gpd.GeoDataFrame:
    """
    Synthetic soil data generation — used ONLY when SoilGrids WCS is down.

    Generates spatially varying clay/bulk_density values based on realistic
    Midwest agricultural soil profiles. NOT real measurements.

    *** MANIFEST WILL SAY SYNTHETIC ***
    """
    logger.warning("=" * 60)
    logger.warning("SOILGRIDS WCS UNAVAILABLE — GENERATING SYNTHETIC SOIL DATA")
    logger.warning("This is NOT real soil measurement data.")
    logger.warning("=" * 60)

    global FALLBACK_USED
    FALLBACK_USED = True

    minx, miny, maxx, maxy = boundary.get_bounds()
    # ~250m grid spacing (matching SoilGrids native resolution)
    step = 0.00225

    records = []
    for lon in np.arange(minx, maxx + step, step):
        for lat in np.arange(miny, maxy + step, step):
            pt = Point(lon, lat)
            # Spatial gradient: clay increases toward SW, bulk density toward E
            norm_x = (lon - minx) / (maxx - minx + 1e-9)
            norm_y = (lat - miny) / (maxy - miny + 1e-9)

            clay = 25.0 + 15.0 * (1.0 - norm_x) * (1.0 - norm_y) + np.random.normal(0, 2)
            clay = float(np.clip(clay, 5.0, 65.0))

            bd = 1.35 + 0.25 * norm_x + 0.10 * norm_y + np.random.normal(0, 0.05)
            bd = float(np.clip(bd, 1.05, 1.95))

            records.append({
                "clay_pct": np.float32(clay),
                "bulk_density": np.float32(bd),
                "geometry": pt,
            })

    if not records:
        records.append({
            "clay_pct": np.float32(30.0),
            "bulk_density": np.float32(1.45),
            "geometry": boundary.geometry.centroid,
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    logger.info("Synthetic soil: %d cells generated", len(gdf))
    return gdf


def _fetch_soilgrids_sync(boundary: FieldBoundary) -> gpd.GeoDataFrame:
    """
    Synchronous WCS fetch. Retrieves INT16 GeoTIFFs, applies CORRECT unit
    scaling, and returns a point-grid GeoDataFrame.
    """
    bbox = boundary.get_bounds()

    # Connectivity check
    url_clay = SOILGRIDS_WCS_TEMPLATE.format(property="clay")
    url_bdod = SOILGRIDS_WCS_TEMPLATE.format(property="bdod")

    try:
        urllib.request.urlopen(
            f"{url_clay}&SERVICE=WCS&REQUEST=GetCapabilities", timeout=15
        )
        logger.info("SoilGrids WCS connectivity confirmed")
    except Exception as e:
        raise PipelineError(f"SoilGrids WCS offline or unreachable: {e}")

    wcs_clay = WebCoverageService(url_clay, version="1.0.0")
    wcs_bdod = WebCoverageService(url_bdod, version="1.0.0")

    wcs_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])

    # Fetch Clay (raw: g/kg mapped value as INT16)
    logger.info("Fetching WCS Coverage: %s", COVERAGEID_CLAY)
    raw_clay = wcs_clay.getCoverage(
        identifier=COVERAGEID_CLAY,
        bbox=wcs_bbox,
        resx=0.002, resy=0.002,
        format=FORMAT,
        crs="EPSG:4326",
    )

    # Fetch Bulk Density (raw: cg/cm3 as INT16)
    logger.info("Fetching WCS Coverage: %s", COVERAGEID_BDOD)
    raw_bdod = wcs_bdod.getCoverage(
        identifier=COVERAGEID_BDOD,
        bbox=wcs_bbox,
        resx=0.002, resy=0.002,
        format=FORMAT,
        crs="EPSG:4326",
    )

    records = []

    with MemoryFile(raw_clay.read()) as mf_clay, MemoryFile(raw_bdod.read()) as mf_bdod:
        with mf_clay.open() as src_clay, mf_bdod.open() as src_bdod:

            clay_data = src_clay.read(1)
            bdod_data = src_bdod.read(1)

            if src_clay.transform != src_bdod.transform:
                logger.warning("WCS grid misalignment between clay and bdod rasters")

            height, width = clay_data.shape
            logger.info("WCS raster shape: %d x %d", height, width)

            for row in range(height):
                for col in range(width):
                    lon, lat = src_clay.xy(row, col)
                    pt = Point(lon, lat)

                    cl_val = clay_data[row, col]
                    bd_val = bdod_data[row, col]

                    # Skip NoData (typically -32768 or similar negatives)
                    if cl_val < 0 or bd_val < 0:
                        continue

                    # =============================================
                    # CRITICAL UNIT SCALING
                    # =============================================
                    # SoilGrids clay: "mapped value" in g/kg
                    #   To get %: divide by 10
                    #   Example: 250 (g/kg) -> 25.0%
                    clay_pct = float(cl_val) / 10.0

                    # SoilGrids bdod: cg/cm3 (centigrams per cubic cm)
                    #   To get g/cm3: divide by 100
                    #   Example: 145 (cg/cm3) -> 1.45 g/cm3
                    bulk_density = float(bd_val) / 100.0

                    records.append({
                        "clay_pct": np.float32(clay_pct),
                        "bulk_density": np.float32(bulk_density),
                        "geometry": pt,
                    })

    if not records:
        raise PipelineError("WCS coverage intersection returned empty grid")

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    logger.info(
        "SoilGrids LIVE: %d cells (clay %.1f-%.1f%%, bd %.2f-%.2f g/cm3)",
        len(gdf),
        gdf["clay_pct"].min(), gdf["clay_pct"].max(),
        gdf["bulk_density"].min(), gdf["bulk_density"].max(),
    )
    return gdf


async def fetch_soilgrids_wcs(boundary: FieldBoundary) -> gpd.GeoDataFrame:
    """
    Wrapper to run blocking WCS calls in async thread pool.
    Falls back to synthetic ONLY on real failure — logged transparently.
    """
    try:
        gdf = await asyncio.to_thread(_fetch_soilgrids_sync, boundary)
        logger.info("SoilGrids returned LIVE data: %d cells", len(gdf))
        return gdf
    except Exception as e:
        logger.error("SoilGrids WCS FAILED: %s", e)
        logger.error("Falling back to synthetic soil generation")
        return _build_synthetic_soil(boundary)
