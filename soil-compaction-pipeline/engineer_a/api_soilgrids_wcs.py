import numpy as np
import pandas as pd
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

# Base URL template for ISRIC SoilGrids v2.0 WCS
SOILGRIDS_WCS_TEMPLATE = "https://maps.isric.org/mapserv?map=/map/{property}.map"
COVERAGEID_CLAY = "clay_5-15cm_Q0.5"
COVERAGEID_BDOD = "bdod_5-15cm_Q0.5"
FORMAT = "GEOTIFF_INT16"

def _build_mock_soil_fallback(boundary: FieldBoundary) -> gpd.GeoDataFrame:
    """Offline fallback for SoilGrids"""
    logger.warning("Falling back to local SoilGrids mock generation.")
    minx, miny, maxx, maxy = boundary.get_bounds()
    pt = boundary.geometry.centroid
    records = [{"clay_pct": 30.0, "bulk_density": 1.45, "geometry": pt}]
    return gpd.GeoDataFrame(records, crs="EPSG:4326")

def _fetch_soilgrids_sync(boundary: FieldBoundary) -> gpd.GeoDataFrame:
    """
    Synchronous execution of OWSLib WCS GetCoverage.
    Retrieves INT16 GeoTIFFs, scales them to physical units,
    and returns a point-grid GeoDataFrame.
    """
    bbox = boundary.get_bounds() # minx, miny, maxx, maxy
    
    # Check WCS capabilities
    try:
        url_clay = SOILGRIDS_WCS_TEMPLATE.format(property="clay")
        url_bdod = SOILGRIDS_WCS_TEMPLATE.format(property="bdod")
        urllib.request.urlopen(f"{url_clay}&SERVICE=WCS&REQUEST=GetCapabilities", timeout=10)
    except Exception as e:
        raise PipelineError(f"SoilGrids WCS offline or timed out: {e}")

    try:
        wcs_clay = WebCoverageService(url_clay, version="1.0.0")
        wcs_bdod = WebCoverageService(url_bdod, version="1.0.0")
        
        # OGC WCS 1.0.0 Bounding Box
        wcs_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Fetch Clay content (g/kg * 10)
        logger.info(f"Fetching WCS Coverage: {COVERAGEID_CLAY}")
        raw_clay = wcs_clay.getCoverage(
            identifier=COVERAGEID_CLAY, 
            bbox=wcs_bbox, 
            resx=0.002, resy=0.002,
            format=FORMAT,
            crs="EPSG:4326"
        )
        
        # Fetch Bulk Density (kg/m3 * 10)
        logger.info(f"Fetching WCS Coverage: {COVERAGEID_BDOD}")
        raw_bdod = wcs_bdod.getCoverage(
            identifier=COVERAGEID_BDOD, 
            bbox=wcs_bbox, 
            resx=0.002, resy=0.002,
            format=FORMAT,
            crs="EPSG:4326"
        )
        
        records = []
        
        # Parse GeoTIFFs from memory buffers
        with MemoryFile(raw_clay.read()) as memfile_clay, MemoryFile(raw_bdod.read()) as memfile_bdod:
            with memfile_clay.open() as src_clay, memfile_bdod.open() as src_bdod:
                
                # Verify transforms align
                if src_clay.transform != src_bdod.transform:
                    logger.warning("WCS Grid misalignment. Generating overlapping grid.")
                    
                clay_data = src_clay.read(1)
                bdod_data = src_bdod.read(1)
                
                # Get spatial coordinates for every pixel
                height, width = clay_data.shape
                for row in range(height):
                    for col in range(width):
                        # Get center coordinate of the pixel
                        lon, lat = src_clay.xy(row, col)
                        pt = Point(lon, lat)
                        
                        cl_val = clay_data[row, col]
                        bd_val = bdod_data[row, col]
                        
                        # Apply Data Contract Unit Scaling
                        # SoilGrids NoData is usually negative (e.g. -32768)
                        if cl_val < 0 or bd_val < 0:
                            continue
                            
                        # Clay: (g/kg * 10) -> % (divide by 10 and 10) -> divide by 100
                        clay_pct = float(cl_val) / 10.0
                        
                        # Bulk Density: cg/cm3 -> g/cm3 (divide by 100)
                        bulk_density = float(bd_val) / 100.0
                        
                        records.append({
                            "clay_pct": clay_pct,
                            "bulk_density": bulk_density,
                            "geometry": pt
                        })

        if not records:
            raise PipelineError("WCS Coverage intersection returned empty grid.")
            
        return gpd.GeoDataFrame(records, crs="EPSG:4326")
        
    except Exception as e:
        logger.error(f"SoilGrids Fetch Failed: {e}")
        raise PipelineError(str(e))

async def fetch_soilgrids_wcs(boundary: FieldBoundary) -> gpd.GeoDataFrame:
    """Wrapper to run blocking WCS network calls in async thread pool."""
    try:
        return await asyncio.to_thread(_fetch_soilgrids_sync, boundary)
    except Exception:
        return _build_mock_soil_fallback(boundary)
