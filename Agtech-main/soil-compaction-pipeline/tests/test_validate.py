import pytest
import pandas as pd
import numpy as np
import logging
from engineer_a.validate import validate_dataframe
from engineer_b.constants import GLOBAL_BOUNDS

def test_validate_all_valid():
    df = pd.DataFrame([{
        "ndvi": 0.5,
        "clay_pct": 25.0,
        "bulk_density": 1.4,
        "equipment_weight_kg": 10000.0,
        "tire_width_m": 0.8
    }])
    res = validate_dataframe(df)
    assert res["data_valid"].all() == True

def test_validate_out_of_bounds():
    df = pd.DataFrame([{
        "ndvi": 2.5, # Out of range [-1, 1]
        "clay_pct": 25.0,
        "bulk_density": 1.4,
        "equipment_weight_kg": 10000.0,
        "tire_width_m": 0.8
    }])
    res = validate_dataframe(df)
    assert res["data_valid"].iloc[0] == False
    assert "ndvi" in res["invalid_fields"].iloc[0]

def test_validate_nan_ignored():
    # Only valid values or NaN should pass True
    df = pd.DataFrame([{
        "ndvi": 0.5,
        "clay_pct": np.nan, 
        "bulk_density": 1.4,
        "equipment_weight_kg": np.nan,
        "tire_width_m": np.nan
    }])
    res = validate_dataframe(df)
    assert res["data_valid"].all() == True

    # Out of bounds is invalid
    df.loc[0, "bulk_density"] = 3.5
    res2 = validate_dataframe(df)
    assert res2["data_valid"].iloc[0] == False
    assert "bulk_density" in res2["invalid_fields"].iloc[0]

def test_validate_soilgrids_scaling():
    # SoilGrids unscaled integers (1450 instead of 1.45) should trigger the auto-scale
    df = pd.DataFrame([{
        "ndvi": 0.5,
        "clay_pct": 25.0,
        "bulk_density": 1450.0, # RAW INT
        "equipment_weight_kg": 10000.0,
        "tire_width_m": 0.8
    }])
    res = validate_dataframe(df)
    # The scaling logic should have divided by 1000, bringing it down to 1.45
    assert res["data_valid"].all() == True
    assert res.loc[0, "bulk_density"] == 1.45
