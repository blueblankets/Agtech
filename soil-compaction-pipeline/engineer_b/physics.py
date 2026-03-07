import math

def contact_pressure_pa(clay_pct: float) -> float:
    """
    Estimate contact pressure based on clay percentage.
    Higher clay might support slightly different contact pressures.
    Assume a baseline tire pressure around 150 kPa (150,000 Pa).
    """
    return 150000.0 + (clay_pct * 500.0)

def sohne_stress(z_cm: float, weight_kg: float, tire_width_m: float,
                 bulk_density: float, clay_pct: float) -> float:
    """Söhne (1953) vertical stress propagation."""
    if weight_kg is None or math.isnan(weight_kg):
        return float('nan')
        
    z_m = z_cm / 100.0
    
    pressure_pa = contact_pressure_pa(clay_pct)
    contact_area_m2 = (weight_kg * 9.81) / pressure_pa
    if contact_area_m2 <= 0:
        return 0.0
        
    radius_m = math.sqrt(contact_area_m2 / math.pi)
    
    # Söhne concentration factor k (empirical, bulk_density dependent)
    k = 4.0 if bulk_density < 1.4 else (5.0 if bulk_density < 1.6 else 6.0)
    
    # Maximum stress under the center line
    sigma_pa = (weight_kg * 9.81 / contact_area_m2) * (1 - (z_m / math.sqrt(z_m**2 + radius_m**2))**k)
    return sigma_pa / 1e6 # Pa -> MPa
