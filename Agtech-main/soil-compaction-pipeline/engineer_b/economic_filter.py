import math
from engineer_b.constants import (
    calculate_roi, ROI_TRIGGER_THRESHOLD, NDVI_STRESS_THRESHOLD,
    MIN_COMPACTION_STRESS_MPA,
)

def determine_action(roi: float, mapie_upper: float, ndvi: float,
                     stress_mpa: float = None) -> str:
    """
    Apply economic guardrails with three tiers:
    
    1. If stress is below MIN_COMPACTION_STRESS_MPA → "None" (soil is fine)
    2. If vegetation is stressed (NDVI < 0.70) AND ROI > threshold → tillage
    3. Otherwise → monitor
    """
    # Guard 1: minimum compaction stress — don't recommend tillage on
    # soil that isn't actually compacted, regardless of ROI
    if stress_mpa is not None and not math.isnan(stress_mpa):
        if stress_mpa < MIN_COMPACTION_STRESS_MPA:
            return "None"
    
    # Guard 2: vegetation must show stress signal
    if mapie_upper > 0 and ndvi < NDVI_STRESS_THRESHOLD:
        if roi > ROI_TRIGGER_THRESHOLD:
            return "Targeted Deep Tillage"
        else:
            return "Monitor - Not Economically Viable"
    return "None"
