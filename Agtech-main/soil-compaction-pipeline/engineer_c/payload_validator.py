"""
Engineer C — Payload & GeoJSON Validator

Validates incoming GeoJSON polygons and outgoing pipeline payloads.
"""


def validate_geojson(geojson: dict) -> tuple[bool, str]:
    """
    Validate a GeoJSON polygon input.
    Returns (is_valid, error_message).
    """
    if not isinstance(geojson, dict):
        return False, "Input must be a JSON object"

    # Accept FeatureCollection or bare Geometry
    geometry = None
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if not features:
            return False, "FeatureCollection has no features"
        geometry = features[0].get("geometry")
    elif geojson.get("type") == "Feature":
        geometry = geojson.get("geometry")
    elif geojson.get("type") in ("Polygon", "MultiPolygon"):
        geometry = geojson
    else:
        return False, f"Unsupported GeoJSON type: {geojson.get('type')}"

    if geometry is None:
        return False, "No geometry found in GeoJSON"

    geo_type = geometry.get("type")
    if geo_type not in ("Polygon", "MultiPolygon"):
        return False, f"Geometry must be Polygon or MultiPolygon, got: {geo_type}"

    coords = geometry.get("coordinates", [])
    if not coords:
        return False, "Geometry has no coordinates"

    # For Polygon, coords[0] is the outer ring
    ring = coords[0] if geo_type == "Polygon" else coords[0][0]
    if len(ring) < 4:  # min 3 vertices + closing point
        return False, f"Ring must have ≥3 vertices (got {len(ring) - 1})"

    # Validate coordinate bounds
    for point in ring:
        lon, lat = point[0], point[1]
        if not (-180 <= lon <= 180):
            return False, f"Longitude {lon} out of range [-180, 180]"
        if not (-90 <= lat <= 90):
            return False, f"Latitude {lat} out of range [-90, 90]"

    return True, ""


def validate_payload_record(record: dict) -> bool:
    """Validate a single record from final_payload.json."""
    required = ["pixel_id", "lat", "lon", "action",
                 "pred_ripper_depth_cm", "mapie_lower_bound",
                 "mapie_upper_bound", "roi"]

    for field in required:
        if field not in record:
            return False

    # Coordinate bounds
    lat = record.get("lat")
    lon = record.get("lon")
    if lat is not None and not (-90 <= lat <= 90):
        return False
    if lon is not None and not (-180 <= lon <= 180):
        return False

    return True


def filter_valid_payload(records: list[dict]) -> list[dict]:
    """Filter payload records, keeping only valid ones."""
    return [r for r in records if validate_payload_record(r)]
