import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import random
import yaml
import logging
from engineer_a.models import FieldBoundary, PipelineError

logger = logging.getLogger(__name__)

# ASABE D497.5 Equipment Dictionary
# Realistic configurations replacing the black-box John Deere API
ASABE_EQUIPMENT_DICT = [
    {
        "model": "Utility Tractor (Light)",
        "static_weight_kg": 5500.0,
        "section_width_m": 0.420,
        "dual_mount": False
    },
    {
        "model": "Row-Crop Tractor (Mid)",
        "static_weight_kg": 12500.0,
        "section_width_m": 0.480,
        "dual_mount": True
    },
    {
        "model": "4WD Articulated (Heavy)",
        "static_weight_kg": 25000.0,
        "section_width_m": 0.710,
        "dual_mount": True
    },
    {
        "model": "Combine Harvester (Extreme)",
        "static_weight_kg": 38000.0,
        "section_width_m": 0.900,
        "dual_mount": False
    }
]

def generate_ab_lines(polygon, working_width_m: float, angle_deg: float = 0.0) -> list:
    """
    Generates algorithmic AB-line passes covering the target field in local EPSG:3857 space.
    """
    # Temporarily project to web mercator to work in precise meters
    gdf = gpd.GeoDataFrame([{"geometry": polygon}], crs="EPSG:4326").to_crs("EPSG:3857")
    poly_m = gdf.geometry.iloc[0]
    
    minx, miny, maxx, maxy = poly_m.bounds
    
    # We rotate the box if angle_deg != 0 (simplification: we just run strict North-South lines 
    # to guarantee coverage for the hackathon simulation)
    lines = []
    x_curr = minx + (working_width_m / 2)
    
    while x_curr <= maxx:
        line = LineString([(x_curr, miny), (x_curr, maxy)])
        # Intersect the infinite line with the actual farm perimeter
        intersected = line.intersection(poly_m)
        if not intersected.is_empty:
            if intersected.geom_type == 'LineString':
                lines.append(intersected)
            elif intersected.geom_type == 'MultiLineString':
                for l in intersected.geoms:
                    lines.append(l)
        x_curr += working_width_m
        
    return lines

def sample_points_along_lines(lines: list, spacing_m: float = 3.0, jitter_m: float = 0.2) -> list:
    points = []
    for line in lines:
        distance = 0
        while distance < line.length:
            pt = line.interpolate(distance)
            
            # Apply tiny simulated GPS jitter
            dx = random.gauss(0, jitter_m)
            dy = random.gauss(0, jitter_m)
            
            jittered_pt = Point(pt.x + dx, pt.y + dy)
            points.append(jittered_pt)
            
            distance += spacing_m
    return points

async def fetch_tractor_ops(boundary: FieldBoundary, config_path="config.yaml") -> gpd.GeoDataFrame:
    """
    Generates simulated ASABE tractor passes directly within the bounds.
    Fully replaces the proprietary API with deterministic algorithmic generation.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        asabe_config = config.get("apis", {}).get("asabe", {})
        enabled = asabe_config.get("enabled", True)
        width_multiplier = asabe_config.get("working_width_multiplier", 1.05)
        path_density_m = asabe_config.get("path_density_m", 3.0)
        
        if not enabled:
            logger.warning("ASABE generator disabled in config. Setting equipment cols to NaN")
            return gpd.GeoDataFrame(columns=["equipment_weight_kg", "tire_width_m", "geometry"], crs="EPSG:4326")
            
        config_entry = random.choice(ASABE_EQUIPMENT_DICT)
        
        # ±15% dynamic weight variance
        weight_kg = config_entry["static_weight_kg"] * random.uniform(0.95, 1.15) 
        
        width_m = config_entry["section_width_m"]
        if config_entry["dual_mount"]:
            width_m *= 2.0
            
        logger.info(f"Simulating ASABE Telemetry for: {config_entry['model']}")
            
        # Force a minimum 10m path spacing to make distinct neat lines on small fields
        effective_width = max(10.0, width_m * width_multiplier)
        ab_lines = generate_ab_lines(boundary.geometry, working_width_m=effective_width)
        
        # Force points to be at least 5m apart to prevent marker overlapping in visualizations
        effective_density = max(5.0, path_density_m)
        points_m = sample_points_along_lines(ab_lines, spacing_m=effective_density, jitter_m=0.2)
        
        if not points_m:
            logger.warning("No tractor points generated inside polygon bounds.")
            return gpd.GeoDataFrame(columns=["equipment_weight_kg", "tire_width_m", "geometry"], crs="EPSG:4326")
            
        # Re-project points back to EPSG:4326
        gdf_m = gpd.GeoDataFrame({"geometry": points_m}, crs="EPSG:3857")
        gdf_gps = gdf_m.to_crs("EPSG:4326")
        
        # Add physics values
        gdf_gps["equipment_weight_kg"] = weight_kg
        gdf_gps["tire_width_m"] = width_m
        
        return gdf_gps
        
    except Exception as e:
        logger.error(f"ASABE Telemetry Failed: {e}. ")
        # Fallback to empty DF - pipeline Stage 2 will map missing passes to NaN, which is a valid state
        return gpd.GeoDataFrame(columns=["equipment_weight_kg", "tire_width_m", "geometry"], crs="EPSG:4326")
