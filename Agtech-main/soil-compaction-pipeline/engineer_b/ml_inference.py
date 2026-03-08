import numpy as np
import xgboost as xgb
from mapie.regression import CrossConformalRegressor
import joblib
import os


def train_prototype_model(output_dir: str):
    """
    Trains a prototype model using MAPIE v1.x CrossConformalRegressor.
    """
    np.random.seed(42)
    n_samples = 1000

    ndvi = np.random.uniform(0.2, 0.9, n_samples)
    clay_pct = np.random.uniform(10, 60, n_samples)
    bulk_density = np.random.uniform(1.2, 1.8, n_samples)
    stress_mpa = np.random.uniform(0.0, 5.0, n_samples)

    base_depth = 10.0 + (stress_mpa * 5.0) + ((1.8 - ndvi) * 10) + ((bulk_density - 1.2) * 20)
    y = base_depth + np.random.normal(0, 3, n_samples)
    y = np.clip(y, 0, 60)

    X = np.column_stack([ndvi, clay_pct, bulk_density, stress_mpa])

    xgb_regressor = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )

    mapie_model = CrossConformalRegressor(
        estimator=xgb_regressor,
        confidence_level=0.90,
        cv=5,
    )
    mapie_model.fit_conformalize(X, y)

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        booster = mapie_model.single_estimator_.get_booster()
    except AttributeError:
        booster = xgb_regressor.fit(X, y).get_booster()
        
    booster.save_model(os.path.join(output_dir, "model.ubj"))
    joblib.dump(mapie_model, os.path.join(output_dir, "mapie_model.pkl"))
    print(f"Saved model.ubj and mapie_model.pkl to {output_dir}")


def run_ml_inference(features: list, mapie_model_path: str):
    mapie = joblib.load(mapie_model_path)
    X_infer = np.array([features])

    pred, intervals = mapie.predict_interval(X_infer)

    depth = float(np.clip(pred[0], 0, 60))
    lo = float(intervals[0, 0, 0])
    hi = float(intervals[0, 1, 0])

    if lo > hi:
        lo, hi = hi, lo

    lo = float(np.clip(lo, 0, 60))
    hi = float(np.clip(hi, 0, 60))

    return depth, lo, hi


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    train_prototype_model(out_dir)
