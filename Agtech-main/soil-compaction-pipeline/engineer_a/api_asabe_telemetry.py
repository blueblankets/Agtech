"""
API C — ASABE D497.5 Synthetic Telemetry Generator.

Generates physically plausible tractor GPS breadcrumb telemetry within
a field polygon using equipment specifications from ASABE Standard D497.5
and public OEM tire databooks (Michelin, Firestone, Maxam).

This replaces the deprecated John Deere Ops Center API.
Zero auth, zero hardware ownership, zero cost.

NOTE: This is SIMULATION by design (per the spec doc). It is not "mock data"
in the same sense as API A/B fallbacks — the spec explicitly defines this as
an algorithmic generator that produces physically plausible telemetry.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import random
import yaml
import logging
from engineer_a.models import FieldBoundary, PipelineError

logger = logging.getLogger(__name__)

# ASABE D497.5 Equipment Dictionary
# Realistic configurations from ASABE Standard + public OEM tire databooks
ASABE_EQUIPMENT_DICT = [
    {
        "model": "Compact Utility 25hp",
        "static_weight_kg": 1200.0,
        "section_width_m": 0.241,
        "dual_mount": False,
    },
    {
        "model": "Utility Tractor 45hp",
        "static_weight_kg": 2500.0,
        "section_width_m": 0.315,
        "dual_mount": False,
    },
    {
        "model": "Row-Crop Tractor 150hp",
        "static_weight_kg": 6800.0,
        "section_width_m": 0.384,
        "dual_mount": False,
    },
    {
        "model": "Row-Crop Tractor 220hp",
        "static_weight_kg": 9500.0,
        "section_width_m": 0.438,
        "dual_mount": True,
    },
    {
        "model": "4WD Articulated 360hp",
        "static_weight_kg": 15000.0,
        "section_width_m": 0.527,
        "dual_mount": True,
    },
    {
        "model": "4WD Articulated 520hp",
        "static_weight_kg": 22000.0,
        "section_width_m": 0.710,
        "dual_mount": True,
    },
    {
        "model": "Track Tractor 620hp",
        "static_weight_kg": 28000.0,
        "section_width_m": 0.914,
        "dual_mount": False,
    },
    {
        "model": "Combine Harvester (Heavy)",
        "static_weight_kg": 35000.0,
        "section_width_m": 0.900,
        "dual_mount": False,
    },
]


def generate_ab_lines(polygon, working_width_m: float) -> list:
    """
    Generates parallel AB guidance lines covering the field in EPSG:3857.
    Real tractors follow strict parallel passes — this simulates that exactly.
    """
    gdf = gpd.GeoDataFrame([{"geometry": polygon}], crs="EPSG:4326").to_crs("EPSG:3857")
    poly_m = gdf.geometry.iloc[0]

    minx, miny, maxx, maxy = poly_m.bounds

    lines = []
    x_curr = minx + (working_width_m / 2)

    while x_curr <= maxx:
        line = LineString([(x_curr, miny - 1), (x_curr, maxy + 1)])
        intersected = line.intersection(poly_m)
        if not intersected.is_empty:
            if intersected.geom_type == "LineString":
                lines.append(intersected)
            elif intersected.geom_type == "MultiLineString":
                for seg in intersected.geoms:
                    lines.append(seg)
        x_curr += working_width_m

    return lines


def sample_points_along_lines(lines: list, spacing_m: float = 3.0, jitter_m: float = 0.2) -> list:
    """
    Generate GPS breadcrumb points along AB lines with realistic RTK GPS noise.
    Spacing of 3m simulates ~12 km/h ground speed at 1 Hz GPS logging.
    """
    points = []
    for line in lines:
        distance = 0
        while distance < line.length:
            pt = line.interpolate(distance)
            # RTK GPS jitter: +-2cm typical, we use +-20cm for DGPS
            dx = random.gauss(0, jitter_m)
            dy = random.gauss(0, jitter_m)
            jittered_pt = Point(pt.x + dx, pt.y + dy)
            points.append(jittered_pt)
            distance += spacing_m
    return points


async def fetch_tractor_ops(boundary: FieldBoundary, config_path="config.yaml") -> gpd.GeoDataFrame:
    """
    Generates ASABE-parameterized tractor passes within the field boundary.

    Simulates a realistic field operation: one randomly selected equipment
    type makes full-coverage parallel passes across the field.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        asabe_config = config.get("apis", {}).get("asabe", {})
        enabled = asabe_config.get("enabled", True)
        width_multiplier = asabe_config.get("working_width_multiplier", 1.05)
        path_density_m = asabe_config.get("path_density_m", 3.0)

        if not enabled:
            logger.warning("ASABE generator disabled in config. Equipment cols = NaN.")
            return gpd.GeoDataFrame(
                columns=["equipment_weight_kg", "tire_width_m", "geometry"],
                crs="EPSG:4326",
            )

        config_entry = random.choice(ASABE_EQUIPMENT_DICT)

        # Dynamic weight variance: +-15% (ballast, implements, fuel load, grain)
        weight_kg = config_entry["static_weight_kg"] * random.uniform(0.85, 1.15)

        # Effective tire width (double for dual-mount configurations)
        width_m = config_entry["section_width_m"]
        if config_entry["dual_mount"]:
            width_m *= 2.0

        logger.info("ASABE equipment selected: %s", config_entry["model"])
        logger.info("  Dynamic weight: %.0f kg, tire width: %.3f m", weight_kg, width_m)

        # Working width = implement width, NOT tire width
        # For tillage/spraying, implement width is much wider than tires
        # Use a realistic implement width based on tractor class
        implement_width_m = _estimate_implement_width(config_entry["static_weight_kg"])
        effective_width = implement_width_m * width_multiplier

        logger.info("  Implement width: %.1f m, effective swath: %.1f m", implement_width_m, effective_width)

        ab_lines = generate_ab_lines(boundary.geometry, working_width_m=effective_width)

        if not ab_lines:
            logger.warning("No AB lines generated — field may be too small")
            return gpd.GeoDataFrame(
                columns=["equipment_weight_kg", "tire_width_m", "geometry"],
                crs="EPSG:4326",
            )

        points_m = sample_points_along_lines(ab_lines, spacing_m=path_density_m, jitter_m=0.2)

        if not points_m:
            logger.warning("No tractor points generated inside polygon bounds")
            return gpd.GeoDataFrame(
                columns=["equipment_weight_kg", "tire_width_m", "geometry"],
                crs="EPSG:4326",
            )

        # Re-project points back to EPSG:4326
        gdf_m = gpd.GeoDataFrame({"geometry": points_m}, crs="EPSG:3857")
        gdf_gps = gdf_m.to_crs("EPSG:4326")

        # Clip to polygon boundary
        gdf_gps = gdf_gps[gdf_gps.geometry.within(boundary.geometry)]

        # Add equipment telemetry values
        gdf_gps["equipment_weight_kg"] = np.float32(weight_kg)
        gdf_gps["tire_width_m"] = np.float32(width_m)

        logger.info(
            "ASABE telemetry: %d breadcrumbs (%s, %.0f kg, %.3f m tire)",
            len(gdf_gps), config_entry["model"], weight_kg, width_m,
        )
        return gdf_gps

    except Exception as e:
        logger.error("ASABE Telemetry FAILED: %s", e)
        return gpd.GeoDataFrame(
            columns=["equipment_weight_kg", "tire_width_m", "geometry"],
            crs="EPSG:4326",
        )


def _estimate_implement_width(static_weight_kg: float) -> float:
    """
    Estimate typical implement working width based on tractor power class.
    Real farms match implement size to tractor capability.
    """
    if static_weight_kg < 3000:
        return 3.0   # Small disc or cultivator
    elif static_weight_kg < 8000:
        return 6.0   # Medium chisel plow
    elif static_weight_kg < 15000:
        return 9.0   # Large field cultivator
    elif static_weight_kg < 25000:
        return 12.0  # Wide air seeder / deep ripper
    else:
        return 15.0  # Combine header / wide tillage bar
