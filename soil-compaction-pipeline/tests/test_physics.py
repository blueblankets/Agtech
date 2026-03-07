import math
import sys
import os

# Add parent directory to path to allow importing engineer_b
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engineer_b.physics import sohne_stress

def test_sohne_stress_nan():
    assert math.isnan(sohne_stress(10, float('nan'), 1.0, 1.5, 30.0))
    print("test_sohne_stress_nan passed!")

def test_sohne_stress_values():
    stress_10 = sohne_stress(10.0, 10000.0, 0.8, 1.5, 30.0)
    stress_30 = sohne_stress(30.0, 10000.0, 0.8, 1.5, 30.0)
    
    assert stress_10 > stress_30
    assert stress_10 > 0
    assert stress_30 > 0

    print(f"test_sohne_stress_values passed! stress_10: {stress_10:.4f} MPa, stress_30: {stress_30:.4f} MPa")

if __name__ == "__main__":
    test_sohne_stress_nan()
    test_sohne_stress_values()
