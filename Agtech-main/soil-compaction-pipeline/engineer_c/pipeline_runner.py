"""
Engineer C — Async Pipeline Runner

Manages the full pipeline execution in a background thread so the Flask
server remains responsive. Tracks stage progression for the status API.
"""
import asyncio
import json
import logging
import os
import sys
import threading
import traceback
from datetime import datetime, timezone

# Ensure pipeline root is importable
PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from engineer_a.ingest import ingest_and_align
from engineer_b.main_pipeline import run_model_pipeline, save_final_payload
from engineer_b.llm_insights import generate_insights

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Singleton-ish pipeline executor. Runs the full A→B→LLM pipeline
    in a background thread and exposes status for polling.
    """

    STAGES = ["idle", "ingesting", "modeling", "insights", "complete", "error"]

    def __init__(self, pipeline_data_dir: str, engineer_b_dir: str):
        self.pipeline_data_dir = pipeline_data_dir
        self.engineer_b_dir = engineer_b_dir
        self.status = "idle"
        self.stage_detail = ""
        self.error_message = ""
        self.started_at = None
        self.completed_at = None
        self._thread = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        if self.status in ("ingesting", "modeling", "insights"):
            # Check if the thread is actually alive — if it died (e.g. server
            # restart, crash), reset state so the user isn't stuck.
            if self._thread is None or not self._thread.is_alive():
                logger.warning("Pipeline thread is dead but status was '%s'. Resetting to idle.", self.status)
                with self._lock:
                    self.status = "idle"
                    self.stage_detail = "Previous run interrupted. Ready for new analysis."
                    self._thread = None
                return False
            return True
        return False

    def get_status(self) -> dict:
        with self._lock:
            elapsed = None
            if self.started_at:
                end = self.completed_at or datetime.now(timezone.utc)
                elapsed = round((end - self.started_at).total_seconds(), 1)

            return {
                "status": self.status,
                "stage_detail": self.stage_detail,
                "error": self.error_message,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "elapsed_seconds": elapsed,
            }

    def start(self, geojson: dict):
        """Launch pipeline in background thread."""
        if self.is_running:
            raise RuntimeError("Pipeline is already running")

        with self._lock:
            self.status = "ingesting"
            self.stage_detail = "Starting data ingestion (Engineer A)..."
            self.error_message = ""
            self.started_at = datetime.now(timezone.utc)
            self.completed_at = None

        self._thread = threading.Thread(
            target=self._run_pipeline, args=(geojson,), daemon=True
        )
        self._thread.start()

    def _run_pipeline(self, geojson: dict):
        """Execute the full pipeline sequentially."""
        try:
            # ─── Stage 1-2: Engineer A ───
            self._set("ingesting", "Running Engineer A: NDVI + SoilGrids + Telemetry...")
            config_path = os.path.join(PIPELINE_ROOT, "config.yaml")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            master_df = loop.run_until_complete(
                ingest_and_align(geojson, config_path, self.pipeline_data_dir)
            )
            loop.close()
            logger.info("Engineer A complete: %d pixels", len(master_df))

            # ─── Stage 3-5: Engineer B ───
            self._set("modeling", "Running Engineer B: Physics + ML + Economic Filter...")
            df_out = run_model_pipeline(master_df, self.engineer_b_dir)

            # Save outputs
            parquet_path = os.path.join(self.pipeline_data_dir, "master_df.parquet")
            df_out.to_parquet(parquet_path, engine="pyarrow", index=False)
            save_final_payload(df_out, self.pipeline_data_dir)
            logger.info("Engineer B complete: %d pixels processed", len(df_out))

            # ─── Stage 6: LLM Insights ───
            self._set("insights", "Running LLM analysis (Gemini 2.5 Flash)...")
            manifest_path = os.path.join(self.pipeline_data_dir, "manifest.json")
            generate_insights(df_out, self.pipeline_data_dir, manifest_path)
            logger.info("LLM insights generated")

            # ─── Done ───
            with self._lock:
                self.status = "complete"
                self.stage_detail = "Pipeline complete"
                self.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            logger.error("Pipeline failed: %s", traceback.format_exc())
            with self._lock:
                self.status = "error"
                self.error_message = str(e)
                self.stage_detail = f"Failed: {e}"
                self.completed_at = datetime.now(timezone.utc)

    def _set(self, status: str, detail: str):
        with self._lock:
            self.status = status
            self.stage_detail = detail
        logger.info("[%s] %s", status.upper(), detail)
