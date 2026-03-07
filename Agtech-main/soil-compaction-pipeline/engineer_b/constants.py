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
TILLAGE_COST_PER_ACRE = 20.0 # USD
COMPACTION_LOSS_PER_ACRE = 30.0 # USD yield loss estimate

def calculate_roi(stress_mpa: float, 
                  tillage_cost: float = TILLAGE_COST_PER_ACRE, 
                  compaction_loss: float = COMPACTION_LOSS_PER_ACRE) -> float:
    """
    Linear scaling: stress 0 -> 1 MPa = 0% loss; 5 MPa = 100% loss.
    ROI = avoided_loss / tillage_cost
    """
    if stress_mpa is None or math.isnan(stress_mpa):
        return 0.0
    
    # "stress >1.5 MPa -> compaction cost $30/acre" as an example in prompt, 
    # but the reference implementation says linearly scaled.
    loss_fraction = min(stress_mpa / 5.0, 1.0)
    avoided_loss = loss_fraction * compaction_loss
    return avoided_loss / tillage_cost if tillage_cost > 0 else 0.0
