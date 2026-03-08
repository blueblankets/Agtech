"""Quick diagnostic: run Eng A pipeline and inspect what happens to ndvi after joins."""
import asyncio, logging, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from engineer_a.ingest import ingest_and_align

GEOJSON = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[
            [-93.593740, 41.516540], [-93.590000, 41.516540],
            [-93.590000, 41.513000], [-93.593740, 41.513000],
            [-93.593740, 41.516540]
        ]]}
    }]
}

async def main():
    try:
        df = await ingest_and_align(GEOJSON, config_path="config.yaml", out_dir="pipeline_data")
        print("\n=== MASTER DF AFTER VALIDATION ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nNDVI stats:")
        print(f"  Total rows: {len(df)}")
        print(f"  NDVI NaN count: {df['ndvi'].isna().sum()}")
        print(f"  NDVI non-NaN count: {df['ndvi'].notna().sum()}")
        print(f"  NDVI min: {df['ndvi'].min()}, max: {df['ndvi'].max()}")
        print(f"\ndata_valid stats:")
        print(f"  Valid: {df['data_valid'].sum()}")
        print(f"  Invalid: {(~df['data_valid']).sum()}")
        print(f"\ninvalid_fields value counts:")
        invalid_only = df[~df['data_valid']]
        if len(invalid_only) > 0:
            print(invalid_only['invalid_fields'].value_counts().head(10))
        print(f"\nFirst 5 rows:")
        print(df.head())
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
