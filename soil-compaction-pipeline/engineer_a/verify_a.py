import asyncio
import json
import os
import folium
import geopandas as gpd
import pandas as pd
from engineer_a.ingest import ingest_and_align
import logging

logging.basicConfig(level=logging.INFO)

# A sample polygon representing a small farm field in Iowa
SAMPLE_GEOJSON = {
  "type": "FeatureCollection",
  "features": [
    {
  "type": "Feature",
  "properties": {},
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [
          -88.378082,
          39.99652
        ],
        [
          -88.377996,
          39.981888
        ],
        [
          -88.368465,
          39.98202
        ],
        [
          -88.368594,
          39.99652
        ],
        [
          -88.378082,
          39.99652
        ]
      ]
    ]
  }
}
  ]
}

def generate_maps(df: pd.DataFrame, out_dir: str):
    # Convert output back to GeoDataFrame for Folium plotting
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )
    
    # Calculate map center
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()
    
    # 1. NDVI Map
    m_ndvi = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles='Esri.WorldImagery')
    for idx, row in gdf.iterrows():
        color = 'green' if row['ndvi'] > 0.6 else ('yellow' if row['ndvi'] > 0.3 else 'red')
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"NDVI: {row['ndvi']:.2f}"
        ).add_to(m_ndvi)
    m_ndvi.save(os.path.join(out_dir, "map_ndvi.html"))

    # 2. Clay Percentage Map
    m_clay = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles='Esri.WorldImagery')
    for idx, row in gdf.iterrows():
        # Darker brown for higher clay
        if pd.isna(row['clay_pct']): continue
        color = '#8B4513' if row['clay_pct'] > 50 else '#DEB887'
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"Clay: {row['clay_pct']:.1f}%"
        ).add_to(m_clay)
    m_clay.save(os.path.join(out_dir, "map_clay.html"))

    # 3. Tractor Equipment Map
    m_tractor = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles='Esri.WorldImagery')
    for idx, row in gdf.iterrows():
        if pd.isna(row['equipment_weight_kg']): continue
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.9,
            popup=f"Weight: {row['equipment_weight_kg']} kg"
        ).add_to(m_tractor)
    m_tractor.save(os.path.join(out_dir, "map_tractor.html"))

    print(f"Maps saved to {out_dir}")

async def main():
    print("--- Starting Engineer A Pipeline Verification ---")
    
    # Run the full ingestion pipeline asynchronously
    df = await ingest_and_align(SAMPLE_GEOJSON, config_path="config.yaml")
    
    print("\n--- Pipeline Output Verified ---")
    print(f"Master DataFrame Head (Total Rows: {len(df)}):")
    cols_to_print = ['pixel_id', 'ndvi', 'clay_pct', 'bulk_density', 'equipment_weight_kg']
    print(df[cols_to_print].head())
    
    # Generate Visualizations
    out_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    generate_maps(df, out_dir)

if __name__ == "__main__":
    asyncio.run(main())
