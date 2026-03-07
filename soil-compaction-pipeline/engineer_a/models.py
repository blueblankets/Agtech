import json
from dataclasses import dataclass
from typing import Dict, Any
import shapely.geometry
import geopandas as gpd

class PipelineError(Exception):
    pass

@dataclass
class FieldBoundary:
    raw_geojson: Dict[str, Any]
    geometry: shapely.geometry.Polygon
    gdf: gpd.GeoDataFrame

    @classmethod
    def from_geojson(cls, geojson_data: dict) -> "FieldBoundary":
        """
        Validates incoming RFC 7946 GeoJSON Polygon.
        Ensures EPSG:4326.
        """
        try:
            if geojson_data.get("type") == "FeatureCollection":
                features = geojson_data.get("features", [])
                if not features:
                    raise ValueError("Empty FeatureCollection")
                geom_dict = features[0].get("geometry")
            elif geojson_data.get("type") == "Feature":
                geom_dict = geojson_data.get("geometry")
            else:
                geom_dict = geojson_data
                
            if geom_dict.get("type") not in ("Polygon", "MultiPolygon"):
                raise ValueError(f"GeoJSON must be Polygon or MultiPolygon. Got: {geom_dict.get('type')}")
                
            if "coordinates" not in geom_dict:
                raise ValueError("GeoJSON missing coordinates key")

            shape = shapely.geometry.shape(geom_dict)
            if not shape.is_valid:
                shape = shape.buffer(0) # Attempt self-intersecting fix
                if not shape.is_valid:
                    raise ValueError("Invalid Polygon Geometry")
            
            gdf = gpd.GeoDataFrame([{"geometry": shape}], crs="EPSG:4326")
            
            return cls(raw_geojson=geojson_data, geometry=shape, gdf=gdf)
            
        except Exception as e:
            raise ValueError(f"boundary parse failed: {str(e)}")

    def get_bounds(self):
        """Returns (minx, miny, maxx, maxy) in EPSG:4326"""
        return self.geometry.bounds

    def to_wkt(self):
        return self.geometry.wkt
