from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pysd import read_vensim
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Vensim System Dynamics Model.mdl"
OUTPUT_CSV = ROOT / "calibration_best_run.csv"
OUTPUT_JSON = ROOT / "calibration_best_parameters.json"

# Update these names only if your Vensim variable names differ.
PARAM_BOUNDS = {
    "infil c": (0.1, 1.0),
    "k perc": (0.01, 0.5),
    "k et": (0.01, 0.2),
    "r": (0.1, 1.5),
    "g0": (0.0, 0.1),
    "d0": (0.0, 0.2),
    "d1": (0.0, 1.0),
    "S half": (0.1, 1.0),
    "a v": (0.0, 1.0),
    "a p": (0.0, 1.0),
    "b": (0.0, 0.3),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
BOUNDS = [PARAM_BOUNDS[name] for name in PARAM_NAMES]
RETURN_COLUMNS = ["NDVI sim", "NDVI obs"]
RETURN_TIMESTAMPS = list(range(120))


def rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    return float(np.sqrt(np.mean((simulated - observed) ** 2)))


def mean_bias(simulated: np.ndarray, observed: np.ndarray) -> float:
    return float(abs(np.mean(simulated) - np.mean(observed)))


def amplitude_gap(simulated: np.ndarray, observed: np.ndarray) -> float:
    sim_amp = np.max(simulated) - np.min(simulated)
    obs_amp = np.max(observed) - np.min(observed)
    return float(abs(sim_amp - obs_amp))


def build_params(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES, values)}


class ModelCalibrator:
    def __init__(self, model_path: Path):
        self.model = read_vensim(str(model_path))

    def run_model(self, params: dict[str, float]) -> pd.DataFrame:
        if hasattr(self.model, "reload"):
            self.model.reload()
        result = self.model.run(
            params=params,
            return_columns=RETURN_COLUMNS,
            return_timestamps=RETURN_TIMESTAMPS,
        )
        return result.reset_index(drop=False)

    def objective(self, values: np.ndarray) -> float:
        params = build_params(values)
        try:
            result = self.run_model(params)
            simulated = result["NDVI sim"].to_numpy(dtype=float)
            observed = result["NDVI obs"].to_numpy(dtype=float)
        except Exception:
            return 1e6

        if not np.isfinite(simulated).all() or not np.isfinite(observed).all():
            return 1e6

        score = (
            rmse(simulated, observed)
            + 0.5 * mean_bias(simulated, observed)
            + 0.5 * amplitude_gap(simulated, observed)
        )
        return float(score)


def main() -> None:
    calibrator = ModelCalibrator(MODEL_PATH)
    result = differential_evolution(
        calibrator.objective,
        bounds=BOUNDS,
        strategy="best1bin",
        maxiter=300,
        popsize=15,
        tol=1e-7,
        polish=True,
        seed=42,
        updating="deferred",
        workers=1,
    )

    best_params = build_params(result.x)
    best_run = calibrator.run_model(best_params)

    simulated = best_run["NDVI sim"].to_numpy(dtype=float)
    observed = best_run["NDVI obs"].to_numpy(dtype=float)

    metrics = {
        "objective": float(result.fun),
        "rmse": rmse(simulated, observed),
        "mean_bias": mean_bias(simulated, observed),
        "amplitude_gap": amplitude_gap(simulated, observed),
    }

    payload = {
        "best_parameters": best_params,
        "metrics": metrics,
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
    }

    best_run.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Calibration finished.")
    print("Best parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value:.6f}")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"Saved best run to: {OUTPUT_CSV}")
    print(f"Saved parameter summary to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
