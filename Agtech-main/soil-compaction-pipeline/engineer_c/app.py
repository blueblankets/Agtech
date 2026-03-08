"""
Engineer C — Flask Application Server

Serves the frontend SPA and provides API endpoints for pipeline
orchestration, status polling, result retrieval, and shapefile export.
"""
import json
import logging
import os
import sys

from flask import Flask, jsonify, request, render_template, send_file

# Ensure pipeline root is importable
PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from engineer_c.pipeline_runner import PipelineRunner
from engineer_c.payload_validator import validate_geojson, filter_valid_payload
from engineer_c.shapefile_export import generate_shapefile_zip

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("engineer_c")

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))

# Pipeline configuration
PIPELINE_DATA_DIR = os.path.join(PIPELINE_ROOT, "pipeline_data")
ENGINEER_B_DIR = os.path.join(PIPELINE_ROOT, "engineer_b")

runner = PipelineRunner(PIPELINE_DATA_DIR, ENGINEER_B_DIR)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main application page."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Accept GeoJSON polygon and trigger the pipeline."""
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    # Validate GeoJSON
    valid, error_msg = validate_geojson(data)
    if not valid:
        return jsonify({"error": f"Invalid GeoJSON: {error_msg}"}), 400

    # Check if already running
    if runner.is_running:
        return jsonify({"error": "Pipeline is already running"}), 409

    # Launch pipeline
    try:
        runner.start(data)
        return jsonify({"status": "started", "message": "Pipeline launched"})
    except Exception as e:
        logger.error("Failed to start pipeline: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def status():
    """Return current pipeline status for polling."""
    return jsonify(runner.get_status())


@app.route("/api/results")
def results():
    """Return pipeline results: payload + insights + summary."""
    payload_path = os.path.join(PIPELINE_DATA_DIR, "final_payload.json")
    insights_path = os.path.join(PIPELINE_DATA_DIR, "insights.json")
    manifest_path = os.path.join(PIPELINE_DATA_DIR, "manifest.json")

    if not os.path.exists(payload_path):
        return jsonify({"error": "No results available. Run the pipeline first."}), 404

    # Load payload
    with open(payload_path, "r") as f:
        payload = json.load(f)

    # Filter invalid records
    payload = filter_valid_payload(payload)

    # Load insights (optional)
    insights = {}
    if os.path.exists(insights_path):
        with open(insights_path, "r") as f:
            insights = json.load(f)

    # Load manifest (optional)
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    # Compute summary statistics
    summary = _compute_summary(payload)

    return jsonify({
        "payload": payload,
        "insights": insights,
        "manifest": manifest,
        "summary": summary,
    })


@app.route("/api/export")
def export():
    """Generate and download prescription shapefile zip."""
    payload_path = os.path.join(PIPELINE_DATA_DIR, "final_payload.json")
    if not os.path.exists(payload_path):
        return jsonify({"error": "No results available"}), 404

    with open(payload_path, "r") as f:
        payload = json.load(f)

    payload = filter_valid_payload(payload)

    zip_path = os.path.join(PIPELINE_DATA_DIR, "prescription.zip")
    generate_shapefile_zip(payload, zip_path)

    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name="prescription.zip",
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

PIXEL_ACRES = 0.0247  # 10m x 10m in acres


def _compute_summary(payload: list[dict]) -> dict:
    """Compute summary statistics grouped by action."""
    total = len(payload)
    if total == 0:
        return {"total_pixels": 0, "total_acreage": 0, "actions": []}

    action_counts = {}
    for record in payload:
        action = record.get("action", "Unknown")
        action_counts[action] = action_counts.get(action, 0) + 1

    actions = []
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        actions.append({
            "action": action,
            "count": count,
            "acreage": round(count * PIXEL_ACRES, 2),
            "percentage": round(count / total * 100, 1),
        })

    return {
        "total_pixels": total,
        "total_acreage": round(total * PIXEL_ACRES, 2),
        "actions": actions,
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🌾 Field Analyzer — Soil Compaction Pipeline")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
