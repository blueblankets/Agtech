"""
LLM Insights Module — Stage 6
Consumes the finalized master_df (after Engineer B processing) and generates
natural-language agronomic insights via an LLM API.

Only AGGREGATED statistics are sent to the LLM (not raw pixels).
Supports OpenAI (GPT-4o) and Google Gemini, with automatic fallback.
"""
import json
import os
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

# Load .env from the project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _v:  # only set if value is non-empty
                    os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert agronomic advisor specializing in soil compaction diagnosis \
and remediation. You have deep expertise in mechanical tillage physics, \
biological root architectures, and soil chemistry amendments.

You will receive a JSON object with aggregated field statistics from a \
geospatial ML pipeline that detects soil compaction using Söhne (1953) \
vertical stress propagation, satellite NDVI, and XGBoost-predicted ripper depths.

YOUR RESPONSE MUST BE HIGHLY PRESCRIPTIVE AND ACTIONABLE. Do not give vague \
advice. Tell the farmer EXACTLY what to do, what equipment to use, at what \
depth, and what amendments to apply.

Use the following agronomic knowledge to inform your recommendations:

## Mechanical Tillage Selection
- If stress > 1.5 MPa AND conservation tillage is desired: prescribe a \
  SUBSOILER with PARABOLIC (bentleg) shanks. Parabolic shanks reduce wheel \
  slippage by ~43% vs straight shanks and create wider subsurface fracturing \
  with minimal topsoil disturbance. Operating depth up to 24 inches (60 cm).
- If stress > 1.5 MPa AND field has rocks or heavy residue requiring \
  incorporation: prescribe a DEEP RIPPER at 14-18 inches (35-45 cm). \
  Expect high topsoil disturbance.
- Depth calibration rule: set the shank tip EXACTLY 1-2 inches (2.5-5 cm) \
  below the bottom of the compaction layer (use pred_ripper_depth_cm from \
  the pipeline). Going deeper wastes fuel; going shallower fails to fracture.
- If subsurface drainage tiles exist, orient passes at an oblique angle to \
  the tile lines to prevent underground pooling.

## Soil Amendments (based on clay_pct and bulk_density)
- If clay_pct > 40% AND sodium is likely elevated (arid/coastal regions): \
  prescribe GYPSUM (CaSO4) at 1-2 tons/acre (max 5 tons/acre/year). \
  Calcium displaces sodium, causing flocculation and structural porosity.
- If clay_pct < 40% OR non-sodic profile: gypsum is CONTRAINDICATED. \
  Instead prescribe HUMIC ACID at 20-40 lbs/acre (granular) or liquid \
  humates into the tillage zone. Humic acids form humic-clay complexes \
  that increase aggregate stability, water holding capacity, and CEC \
  while lowering the soil's modulus of rupture.
- If bulk_density > 1.6 g/cm³: the soil is severely compacted. \
  Mechanical intervention is almost certainly required before amendments \
  can penetrate effectively.

## Biological Drilling (Cover Crop Cocktail)
After any mechanical subsoiling, IMMEDIATELY seed a multi-species cover \
crop cocktail to biologically stabilize the fractured profile:
- TAPROOTS (mandatory): Daikon/tillage radish, turnips, or rapeseed. \
  Their massive taproots drill into fractured subsoil fissures, creating \
  permanent biopores. Upon winter-kill, decomposing roots leave open \
  channels for spring cash crop root access.
- FIBROUS ROOTS (mandatory): Cereal rye, oats, or annual ryegrass. \
  Dense branching roots exude glomalin that cements topsoil aggregates, \
  eliminating surface crusting and stabilizing disturbed soil.
- LEGUMES (recommended): Crimson clover or hairy vetch. Fix 50-150 lbs \
  N/acre via rhizobia, lowering C:N ratio and accelerating organic matter \
  formation. Higher organic matter = higher bearing capacity = resistance \
  to future compaction.

## NDVI Interpretation
- NDVI < 0.30: Severe vegetation stress, likely root restriction from \
  compaction or nutrient lockout. Investigate immediately.
- NDVI 0.30-0.50: Moderate stress, compaction may be limiting root depth \
  and water access. Cross-reference with stress data.
- NDVI 0.50-0.70: Mild stress, monitor but intervention may not be \
  economically justified unless ROI > 1.2.
- NDVI > 0.70: Healthy vegetation, compaction is not currently limiting yield.

## Economic Decision Framework
- ROI > 1.2: Tillage is economically justified. Proceed with full \
  mechanical + amendment + cover crop protocol.
- ROI 0.8-1.2: Marginal. Consider biological-only intervention (cover \
  crops + amendments, skip mechanical tillage).
- ROI < 0.8: Not economically viable for mechanical tillage. Deploy \
  cover crops only and reassess next season.

Produce a structured JSON response with these keys:

1. "field_summary" — 2-3 sentence overview of field condition.

2. "risk_assessment" — Identify highest-risk zones, explain WHY they are \
   at risk (link stress, NDVI, clay, bulk density).

3. "recommendations" — A list of 3-5 SPECIFIC, PRIORITIZED actions. Each \
   MUST include: exact implement type, operating depth in cm, amendment \
   type and rate, and cover crop species. Reference the pipeline data \
   (e.g., "Based on the mean predicted ripper depth of 30.7 cm, set \
   subsoiler shanks to 33 cm").

4. "economic_outlook" — Economic case for/against intervention with \
   specific dollar estimates per acre where possible.

5. "confidence_note" — One sentence about model uncertainty from MAPIE \
   prediction interval widths.

Respond ONLY with valid JSON. No markdown, no preamble.
"""

# ─── Summary Builder ─────────────────────────────────────────────────────────

def build_field_summary(df: pd.DataFrame, manifest_path: str = None) -> dict:
    """
    Compute aggregated statistics from the finalized master_df.
    Returns a compact dict suitable for LLM consumption (~800 tokens).
    """
    total = len(df)
    valid = int(df["data_valid"].sum()) if "data_valid" in df.columns else total
    invalid = total - valid

    # Action distribution
    action_dist = {}
    if "action" in df.columns:
        action_dist = df["action"].value_counts().to_dict()
        # Convert numpy int64 to regular int for JSON serialization
        action_dist = {k: int(v) for k, v in action_dist.items()}

    # Stress statistics
    stress_col = df["max_subsoil_stress_mpa"].dropna() if "max_subsoil_stress_mpa" in df.columns else pd.Series()
    stress_stats = {}
    if len(stress_col) > 0:
        stress_stats = {
            "mean": round(float(stress_col.mean()), 3),
            "max": round(float(stress_col.max()), 3),
            "p90": round(float(stress_col.quantile(0.90)), 3),
            "p10": round(float(stress_col.quantile(0.10)), 3),
        }

    # NDVI statistics
    ndvi_col = df["ndvi"].dropna() if "ndvi" in df.columns else pd.Series()
    ndvi_stats = {}
    if len(ndvi_col) > 0:
        ndvi_stats = {
            "mean": round(float(ndvi_col.mean()), 3),
            "min": round(float(ndvi_col.min()), 3),
            "max": round(float(ndvi_col.max()), 3),
            "pct_below_070": round(float((ndvi_col < 0.70).mean() * 100), 1),
        }

    # Soil properties
    soil_stats = {}
    if "clay_pct" in df.columns:
        clay = df["clay_pct"].dropna()
        if len(clay) > 0:
            soil_stats["clay_pct_mean"] = round(float(clay.mean()), 1)
            soil_stats["clay_pct_range"] = [round(float(clay.min()), 1), round(float(clay.max()), 1)]
    if "bulk_density" in df.columns:
        bd = df["bulk_density"].dropna()
        if len(bd) > 0:
            soil_stats["bulk_density_mean"] = round(float(bd.mean()), 2)

    # ROI statistics
    roi_col = df["roi"].dropna() if "roi" in df.columns else pd.Series()
    roi_stats = {}
    if len(roi_col) > 0:
        roi_stats = {
            "mean": round(float(roi_col.mean()), 3),
            "max": round(float(roi_col.max()), 3),
            "pct_above_1_2": round(float((roi_col > 1.2).mean() * 100), 1),
        }

    # MAPIE confidence intervals
    confidence = {}
    if "mapie_lower_bound" in df.columns and "mapie_upper_bound" in df.columns:
        lo = df["mapie_lower_bound"].dropna()
        hi = df["mapie_upper_bound"].dropna()
        if len(lo) > 0 and len(hi) > 0:
            widths = hi.values[:len(lo)] - lo.values[:len(hi)]
            confidence = {
                "mean_interval_width_cm": round(float(np.nanmean(widths)), 1),
                "max_interval_width_cm": round(float(np.nanmax(widths)), 1),
            }

    # Ripper depth statistics
    depth_col = df["pred_ripper_depth_cm"].dropna() if "pred_ripper_depth_cm" in df.columns else pd.Series()
    ripper_stats = {}
    if len(depth_col) > 0:
        ripper_stats = {
            "mean": round(float(depth_col.mean()), 1),
            "p10": round(float(depth_col.quantile(0.10)), 1),
            "p90": round(float(depth_col.quantile(0.90)), 1),
        }

    # Data source health from manifest
    api_health = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        api_health = manifest.get("api_health", {})

    return {
        "total_pixels": total,
        "valid_pixels": valid,
        "invalid_pixels": invalid,
        "action_distribution": action_dist,
        "stress_stats": stress_stats,
        "ndvi_stats": ndvi_stats,
        "soil": soil_stats,
        "roi_stats": roi_stats,
        "confidence": confidence,
        "ripper_depth_stats": ripper_stats,
        "data_sources": api_health,
    }


# ─── LLM Callers ─────────────────────────────────────────────────────────────

def _call_openai(summary_json: str) -> dict:
    """Call OpenAI GPT-4o API."""
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here are the field statistics:\n\n{summary_json}"},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    usage = response.usage
    result = json.loads(response.choices[0].message.content)
    result["metadata"] = {
        "model": "gpt-4o",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "token_usage": {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
        },
    }
    return result


def _call_gemini(summary_json: str) -> dict:
    """Call Google Gemini API using the new google-genai SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    models = ["gemini-2.5-flash", "gemini-2.0-flash"]
    last_err = None

    for model_name in models:
        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\nHere are the field statistics:\n\n{summary_json}"
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            result = json.loads(response.text)
            result["metadata"] = {
                "model": model_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            return result
        except Exception as e:
            logger.warning(f"{model_name} failed: {e}")
            last_err = e

    raise last_err


# ─── Main Entry Point ────────────────────────────────────────────────────────

def generate_insights(
    df: pd.DataFrame,
    output_dir: str,
    manifest_path: str = None,
    provider: str = "auto",
) -> dict:
    """
    Generate LLM-powered agronomic insights from the finalized pipeline output.

    Args:
        df: Finalized master_df with all Engineer B columns.
        output_dir: Directory to save insights.json.
        manifest_path: Optional path to manifest.json for data source info.
        provider: "openai", "gemini", or "auto" (tries both with fallback).

    Returns:
        Parsed insights dict (also saved to insights.json).
    """
    # 1. Build summary
    if manifest_path is None:
        manifest_path = os.path.join(output_dir, "manifest.json")

    summary = build_field_summary(df, manifest_path)
    summary_json = json.dumps(summary, indent=2)
    logger.info("Field summary built: %d tokens (approx)", len(summary_json) // 4)

    # 2. Call LLM
    result = None
    errors = []

    providers_to_try = []
    if provider == "auto":
        # Try OpenAI first if key exists, then Gemini
        if os.environ.get("OPENAI_API_KEY"):
            providers_to_try.append("openai")
        if os.environ.get("GOOGLE_API_KEY"):
            providers_to_try.append("gemini")
        if not providers_to_try:
            providers_to_try = ["openai", "gemini"]
    else:
        providers_to_try = [provider]

    for p in providers_to_try:
        try:
            if p == "openai":
                logger.info("Calling OpenAI GPT-4o for insights...")
                result = _call_openai(summary_json)
            elif p == "gemini":
                logger.info("Calling Google Gemini for insights...")
                result = _call_gemini(summary_json)
            logger.info("LLM insights generated via %s", p)
            break
        except Exception as e:
            logger.warning("LLM provider '%s' failed: %s", p, e)
            errors.append(f"{p}: {e}")

    # 3. Fallback if all providers fail
    if result is None:
        logger.error("All LLM providers failed: %s", errors)
        result = _build_fallback_insights(summary)

    # 4. Attach the raw summary for transparency
    result["input_summary"] = summary

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "insights.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Insights saved to %s", out_path)

    return result


def _build_fallback_insights(summary: dict) -> dict:
    """
    Rule-based fallback when no LLM is available.
    Generates basic insights from the statistics directly.
    """
    action_dist = summary.get("action_distribution", {})
    total = summary.get("total_pixels", 0)
    tillage_count = action_dist.get("Targeted Deep Tillage", 0)
    monitor_count = action_dist.get("Monitor - Not Economically Viable", 0)
    none_count = action_dist.get("None", 0)

    tillage_pct = (tillage_count / total * 100) if total > 0 else 0
    stress = summary.get("stress_stats", {})
    ndvi = summary.get("ndvi_stats", {})
    roi = summary.get("roi_stats", {})
    ripper = summary.get("ripper_depth_stats", {})

    field_summary = (
        f"Field contains {total} analyzed pixels. "
        f"{tillage_pct:.0f}% of the field requires targeted deep tillage. "
        f"Mean subsoil stress is {stress.get('mean', 'N/A')} MPa "
        f"with mean NDVI of {ndvi.get('mean', 'N/A')}."
    )

    recommendations = []
    if tillage_pct > 30:
        recommendations.append(
            f"High priority: {tillage_pct:.0f}% of field needs deep ripping "
            f"at approximately {ripper.get('mean', 30):.0f}cm depth."
        )
    if stress.get("max", 0) > 3.0:
        recommendations.append(
            f"Stress hotspots detected (max {stress['max']} MPa). "
            "Consider reducing equipment weight or increasing tire width in these zones."
        )
    if ndvi.get("pct_below_070", 0) > 50:
        recommendations.append(
            f"{ndvi['pct_below_070']:.0f}% of vegetation is below the stress threshold. "
            "Compaction may be limiting root growth and nutrient uptake."
        )
    if not recommendations:
        recommendations.append("Field shows low compaction risk. Continue monitoring.")

    return {
        "field_summary": field_summary,
        "risk_assessment": f"Mean stress: {stress.get('mean', 'N/A')} MPa, "
                          f"P90 stress: {stress.get('p90', 'N/A')} MPa. "
                          f"{ndvi.get('pct_below_070', 0):.0f}% of pixels show NDVI below 0.70.",
        "recommendations": recommendations,
        "economic_outlook": f"Mean ROI is {roi.get('mean', 'N/A')}. "
                           f"{roi.get('pct_above_1_2', 0):.0f}% of field exceeds the 1.2 ROI trigger.",
        "confidence_note": f"Mean prediction interval width: "
                          f"{summary.get('confidence', {}).get('mean_interval_width_cm', 'N/A')} cm.",
        "metadata": {
            "model": "rule-based-fallback",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "LLM unavailable — insights generated from rule-based analysis.",
        },
    }


if __name__ == "__main__":
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_data = os.path.join(base, "pipeline_data")

    parquet_path = os.path.join(pipeline_data, "master_df.parquet")
    if not os.path.exists(parquet_path):
        print(f"ERROR: {parquet_path} not found. Run e2e_test.py first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    insights = generate_insights(df, pipeline_data)
    print(json.dumps(insights, indent=2))
