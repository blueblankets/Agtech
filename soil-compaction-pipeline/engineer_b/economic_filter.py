import math
from engineer_b.constants import calculate_roi, ROI_TRIGGER_THRESHOLD, NDVI_STRESS_THRESHOLD

def determine_action(roi: float, mapie_upper: float, ndvi: float) -> str:
    """
    Apply economic guardrails.
    Guard: if roi > 1.2 AND mapie_upper > 0 AND ndvi < 0.70 -> "Targeted Deep Tillage"
    """
    if mapie_upper > 0 and ndvi < NDVI_STRESS_THRESHOLD:
        if roi > ROI_TRIGGER_THRESHOLD:
            return "Targeted Deep Tillage"
        else:
            return "Monitor - Not Economically Viable"
    return "None"
