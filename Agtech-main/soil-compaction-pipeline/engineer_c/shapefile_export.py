"""
Engineer C — Shapefile Export

Generates a .shp/.shx/.dbf/.prj bundle (prescription.zip) from pipeline output.
Uses pyshp (pure Python, no GDAL dependency).
"""
import io
import json
import math
import os
import zipfile

import shapefile  # pyshp


# DBF field name mapping (max 10 chars)
DBF_FIELDS = {
    "pixel_id":           ("PIXEL_ID",  "C", 10),
    "lat":                ("LAT",       "N", 12, 6),
    "lon":                ("LON",       "N", 12, 6),
    "action":             ("ACTION",    "C", 40),
    "pred_ripper_depth_cm": ("PRED_RIP",  "N", 8, 1),
    "mapie_lower_bound":  ("MAPIE_LO",  "N", 8, 1),
    "mapie_upper_bound":  ("MAPIE_HI",  "N", 8, 1),
    "roi":                ("ROI",       "N", 8, 3),
}

# WGS84 .prj content
WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)


def _pixel_to_polygon(lat: float, lon: float, size_m: float = 10.0):
    """Convert a center lat/lon to a square polygon of given size in meters."""
    half = size_m / 2.0
    lat_offset = half / 111320.0  # degrees per meter
    lon_offset = half / (111320.0 * math.cos(math.radians(lat)))

    return [
        [lon - lon_offset, lat + lat_offset],  # NW
        [lon + lon_offset, lat + lat_offset],  # NE
        [lon + lon_offset, lat - lat_offset],  # SE
        [lon - lon_offset, lat - lat_offset],  # SW
        [lon - lon_offset, lat + lat_offset],  # close
    ]


def generate_shapefile_zip(payload: list[dict], output_path: str = None) -> bytes:
    """
    Generate a prescription.zip containing .shp, .shx, .dbf, .prj files.

    Args:
        payload: list of pixel records from final_payload.json
        output_path: optional file path to write zip to disk

    Returns:
        bytes of the zip file
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "prescription")

        # pyshp 3.x: Writer takes target path as first arg
        w = shapefile.Writer(base)
        w.shapeType = shapefile.POLYGON

        # Define fields
        for orig_name, field_def in DBF_FIELDS.items():
            name = field_def[0]
            ftype = field_def[1]
            if ftype == "C":
                w.field(name, ftype, size=field_def[2])
            else:
                w.field(name, ftype, size=field_def[2], decimal=field_def[3])

        # Write records
        for record in payload:
            lat = record.get("lat")
            lon = record.get("lon")

            if lat is None or lon is None:
                continue

            # Write polygon geometry
            poly = _pixel_to_polygon(lat, lon)
            w.poly([poly])

            # Write attributes
            w.record(
                record.get("pixel_id", ""),
                lat,
                lon,
                record.get("action", ""),
                record.get("pred_ripper_depth_cm") or 0,
                record.get("mapie_lower_bound") or 0,
                record.get("mapie_upper_bound") or 0,
                record.get("roi") or 0,
            )

        w.close()

        # Write .prj
        with open(base + ".prj", "w") as prj_f:
            prj_f.write(WGS84_PRJ)

        # Bundle into zip
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext in [".shp", ".shx", ".dbf", ".prj"]:
                filepath = base + ext
                zf.write(filepath, "prescription" + ext)

        zip_bytes = zip_buf.getvalue()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(zip_bytes)

    return zip_bytes
