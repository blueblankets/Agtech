"""
End-to-End Integration Test: Engineer A → Engineer B → Visualizations

Runs the full pipeline from a sample GeoJSON polygon through:
  Stage 1-2 (Engineer A): API ingestion + spatial alignment → master_df.parquet
  Stage 3-5 (Engineer B): Söhne physics + XGBoost/MAPIE + economic filter → final_payload.json
  Visualization: Heatmaps of all output columns (Engineer B style)
"""
import asyncio
import os
import sys
import logging

# Ensure the pipeline root is on PYTHONPATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engineer_a.ingest import ingest_and_align
from engineer_b.main_pipeline import run_model_pipeline, save_final_payload
from engineer_b.verify_and_visualize import verify_and_visualize
from e2e_visualize import e2e_visualize

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("e2e")

# ─── Sample GeoJSON: ~40-acre Iowa farm polygon ───
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
          -92.853387,
          38.457924
        ],
        [
          -92.853,
          38.450665
        ],
        [
          -92.842353,
          38.451538
        ],
        [
          -92.843727,
          38.457621
        ],
        [
          -92.853387,
          38.457924
        ]
      ]
    ]
  }
}
  ]
}


async def run_e2e():
    pipeline_data_dir = os.path.join(BASE_DIR, "pipeline_data")
    engineer_b_dir = os.path.join(BASE_DIR, "engineer_b")
    viz_dir = os.path.join(BASE_DIR, "e2e_visualizations")

    # ═══════════════════════════════════════════════
    # STAGE 1-2: Engineer A — Data Ingestion & Alignment
    # ═══════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 1-2: ENGINEER A — Data Ingestion & Spatial Alignment")
    logger.info("=" * 60)

    master_df = await ingest_and_align(
        SAMPLE_GEOJSON,
        config_path=os.path.join(BASE_DIR, "config.yaml"),
        out_dir=pipeline_data_dir
    )

    logger.info(f"Engineer A produced {len(master_df)} pixels.")
    logger.info(f"Columns: {list(master_df.columns)}")
    print("\n--- Engineer A Output (first 5 rows) ---")
    print(master_df.head())

    # ═══════════════════════════════════════════════
    # STAGE 3-5: Engineer B — Physics + ML + Economic Filter
    # ═══════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 3-5: ENGINEER B — Physics, ML Inference, Economic Filter")
    logger.info("=" * 60)

    # run_model_pipeline mutates the DataFrame in-place and returns it
    df_out = run_model_pipeline(master_df, engineer_b_dir)

    logger.info(f"Engineer B processed {len(df_out)} pixels.")
    print("\n--- Engineer B Output (first 5 rows) ---")
    print(df_out[["pixel_id", "max_subsoil_stress_mpa", "pred_ripper_depth_cm",
                   "mapie_lower_bound", "mapie_upper_bound", "roi", "action"]].head())

    # Save updated parquet with Engineer B columns included
    parquet_path = os.path.join(pipeline_data_dir, "master_df.parquet")
    df_out.to_parquet(parquet_path, engine="pyarrow", index=False)
    logger.info(f"Updated master_df.parquet with Engineer B columns.")

    # Save final JSON payload
    save_final_payload(df_out, pipeline_data_dir)

    # ═══════════════════════════════════════════════
    # VISUALIZATION: Engineer B-style heatmaps
    # ═══════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("VISUALIZATION: Generating Engineer B heatmaps")
    logger.info("=" * 60)

    payload_path = os.path.join(pipeline_data_dir, "final_payload.json")
    e2e_visualize(parquet_path, payload_path, viz_dir)

    logger.info("=" * 60)
    logger.info("E2E PIPELINE COMPLETE")
    logger.info(f"Heatmaps saved to: {viz_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_e2e())
