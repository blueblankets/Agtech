import math

GLOBAL_BOUNDS = {
    "ndvi": (-1.0, 1.0),
    "clay_pct": (0.0, 100.0),
    "bulk_density": (1.0, 2.0),
    "equipment_weight_kg": (500.0, 40000.0),
    "tire_width_m": (0.1, 2.0),
    "max_subsoil_stress_mpa": (0.0, 5.0),
    "pred_ripper_depth_cm": (0.0, 60.0),
}

VALID_ACTIONS = {
    "Targeted Deep Tillage",
    "Monitor - Not Economically Viable",
    "None",
    "INVALID_DATA",
}

ROI_TRIGGER_THRESHOLD = 1.2 # must exceed to recommend tillage
NDVI_STRESS_THRESHOLD = 0.70 # NDVI below this = potential compaction
TILLAGE_COST_PER_ACRE = 30.0 # USD — deep ripping cost including fuel + labor
COMPACTION_LOSS_PER_ACRE = 120.0 # USD — yield loss from compaction (corn/soy avg)

# Stress level at which full yield loss occurs
# 0.5 MPa is the agronomic threshold for severe compaction
STRESS_FULL_DAMAGE_MPA = 0.5

def calculate_roi(stress_mpa: float, 
                  tillage_cost: float = TILLAGE_COST_PER_ACRE, 
                  compaction_loss: float = COMPACTION_LOSS_PER_ACRE) -> float:
    """
    ROI = (avoided_yield_loss) / tillage_cost
    
    Linear scaling from 0 to STRESS_FULL_DAMAGE_MPA:
    - At 0.15 MPa stress → ROI ≈ 1.2 (decision boundary)
    - At 0.25 MPa stress → ROI ≈ 2.0 (strong tillage signal)
    - At 0.50 MPa stress → ROI = 4.0 (maximum)
    """
    if stress_mpa is None or math.isnan(stress_mpa):
        return 0.0
    
    if stress_mpa <= 0:
        return 0.0
    
    loss_fraction = min(stress_mpa / STRESS_FULL_DAMAGE_MPA, 1.0)
    avoided_loss = loss_fraction * compaction_loss
    return avoided_loss / tillage_cost if tillage_cost > 0 else 0.0
