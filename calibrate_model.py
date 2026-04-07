from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pysd import read_vensim
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Vensim System Dynamics Model.mdl"
OUTPUT_CSV = ROOT / "calibration_best_run.csv"
OUTPUT_JSON = ROOT / "calibration_best_parameters.json"

# Baseline values are the original constants from the Vensim/PySD model.
DEFAULT_PARAMS = {
    "infil c": 0.6,
    "k perc": 0.15,
    "k et": 0.05,
    "r": 0.5,
    "g0": 0.01,
    "d0": 0.08,
    "d1": 0.35,
    "S half": 0.5,
    "a v": 0.1,
    "a p": 0.16,
    "b": 0.09,
}

# Narrower, physically safer bounds keep calibration from finding NDVI-only
# optima that destabilize the hidden stock-flow dynamics once re-used in Vensim.
PARAM_BOUNDS = {
    "infil c": (0.30, 0.90),
    "k perc": (0.05, 0.25),
    "k et": (0.01, 0.12),
    "r": (0.20, 0.70),
    "g0": (0.00, 0.05),
    "d0": (0.03, 0.12),
    "d1": (0.10, 0.45),
    "S half": (0.20, 0.80),
    "a v": (0.05, 0.20),
    "a p": (0.05, 0.25),
    "b": (0.02, 0.15),
}

PARAM_NAMES = list(DEFAULT_PARAMS.keys())
BOUNDS = [PARAM_BOUNDS[name] for name in PARAM_NAMES]
RETURN_TIMESTAMPS = list(range(132))
RETURN_COLUMNS = [
    "NDVI sim",
    "NDVI sim raw",
    "NDVI obs",
    "Soil Water",
    "Vegetation Biomass",
    "Growth",
    "Degradation",
    "Infiltration",
    "Percolation",
    "Evapotranspiration",
]

LARGE_PENALTY = 1e6


def build_params(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES, values)}


def values_from_params(params: dict[str, float]) -> np.ndarray:
    return np.array([params[name] for name in PARAM_NAMES], dtype=float)


def rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    return float(np.sqrt(np.mean((simulated - observed) ** 2)))


def mean_bias(simulated: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean(simulated - observed))


def amplitude_gap(simulated: np.ndarray, observed: np.ndarray) -> float:
    return float((np.max(simulated) - np.min(simulated)) - (np.max(observed) - np.min(observed)))


def soft_bound_penalty(values: np.ndarray, lower: float, upper: float) -> float:
    below = np.clip(lower - values, a_min=0.0, a_max=None)
    above = np.clip(values - upper, a_min=0.0, a_max=None)
    return float(np.mean(below**2 + above**2))


def regularization_penalty(values: np.ndarray) -> float:
    default_values = values_from_params(DEFAULT_PARAMS)
    spans = np.array([high - low for low, high in BOUNDS], dtype=float)
    scaled = (values - default_values) / spans
    return float(np.mean(scaled**2))


def clipping_penalty(values: np.ndarray, tolerance: float = 1e-6) -> float:
    return float(np.mean((values <= tolerance) | (values >= 1 - tolerance)))


def flat_series_penalty(values: np.ndarray, minimum_std: float) -> float:
    std = float(np.std(values))
    if std >= minimum_std:
        return 0.0
    return float((minimum_std - std) ** 2)


def rename_time_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index(drop=False)
    first_col = result.columns[0]
    if first_col != "time":
        result = result.rename(columns={first_col: "time"})
    return result


@dataclass
class CalibrationResult:
    start_index: int
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int


class ModelCalibrator:
    def __init__(self, model_path: Path):
        self.model = read_vensim(str(model_path))

    def run_model(self, params: dict[str, float]) -> pd.DataFrame:
        if hasattr(self.model, "reload"):
            self.model.reload()

        frame = self.model.run(
            params=params,
            return_columns=RETURN_COLUMNS,
            return_timestamps=RETURN_TIMESTAMPS,
        )
        return rename_time_column(frame)

    def evaluate(self, values: np.ndarray) -> tuple[float, dict[str, float]]:
        params = build_params(values)

        try:
            result = self.run_model(params)
        except Exception:
            return LARGE_PENALTY, {"simulation_failure": LARGE_PENALTY}

        ndvi_sim = result["NDVI sim"].to_numpy(dtype=float)
        ndvi_raw = result["NDVI sim raw"].to_numpy(dtype=float)
        ndvi_obs = result["NDVI obs"].to_numpy(dtype=float)
        soil_water = result["Soil Water"].to_numpy(dtype=float)
        vegetation = result["Vegetation Biomass"].to_numpy(dtype=float)

        series = [ndvi_sim, ndvi_raw, ndvi_obs, soil_water, vegetation]
        if any(not np.isfinite(item).all() for item in series):
            return LARGE_PENALTY, {"non_finite_values": LARGE_PENALTY}

        metrics = {
            "rmse": rmse(ndvi_sim, ndvi_obs),
            "mean_bias_sq": mean_bias(ndvi_sim, ndvi_obs) ** 2,
            "amplitude_gap_sq": amplitude_gap(ndvi_sim, ndvi_obs) ** 2,
            "soil_water_out_of_range": soft_bound_penalty(soil_water, 0.0, 1.0),
            "vegetation_out_of_range": soft_bound_penalty(vegetation, 0.0, 1.0),
            "ndvi_raw_out_of_range": soft_bound_penalty(ndvi_raw, 0.0, 1.0),
            "ndvi_clipping": clipping_penalty(ndvi_sim),
            "vegetation_flatness": flat_series_penalty(vegetation, minimum_std=0.01),
            "soil_water_flatness": flat_series_penalty(soil_water, minimum_std=0.02),
            "parameter_regularization": regularization_penalty(values),
        }

        score = (
            metrics["rmse"]
            + 0.30 * metrics["mean_bias_sq"]
            + 0.20 * metrics["amplitude_gap_sq"]
            + 40.0 * metrics["soil_water_out_of_range"]
            + 40.0 * metrics["vegetation_out_of_range"]
            + 25.0 * metrics["ndvi_raw_out_of_range"]
            + 2.5 * metrics["ndvi_clipping"]
            + 2.0 * metrics["vegetation_flatness"]
            + 1.5 * metrics["soil_water_flatness"]
            + 0.35 * metrics["parameter_regularization"]
        )
        metrics["objective"] = float(score)
        return float(score), metrics

    def objective(self, values: np.ndarray) -> float:
        score, _ = self.evaluate(values)
        return float(score)


def generate_start_points(n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [values_from_params(DEFAULT_PARAMS)]
    for _ in range(max(0, n_starts - 1)):
        candidate = np.array(
            [rng.uniform(low, high) for low, high in BOUNDS],
            dtype=float,
        )
        starts.append(candidate)
    return starts


def run_multistart_optimization(
    calibrator: ModelCalibrator,
    n_starts: int = 12,
    seed: int = 42,
) -> tuple[CalibrationResult, list[CalibrationResult]]:
    runs: list[CalibrationResult] = []

    for index, start in enumerate(generate_start_points(n_starts=n_starts, seed=seed), start=1):
        result = minimize(
            calibrator.objective,
            x0=start,
            method="L-BFGS-B",
            bounds=BOUNDS,
            options={
                "maxiter": 400,
                "ftol": 1e-9,
                "maxls": 50,
            },
        )

        runs.append(
            CalibrationResult(
                start_index=index,
                x=np.array(result.x, dtype=float),
                fun=float(result.fun),
                success=bool(result.success),
                message=str(result.message),
                nit=int(getattr(result, "nit", 0)),
                nfev=int(getattr(result, "nfev", 0)),
            )
        )

    best = min(runs, key=lambda item: item.fun)
    return best, runs


def save_outputs(
    best: CalibrationResult,
    all_runs: list[CalibrationResult],
    best_run: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    best_params = build_params(best.x)
    payload: dict[str, Any] = {
        "method": {
            "optimizer": "scipy.optimize.minimize",
            "algorithm": "L-BFGS-B",
            "multistart_runs": len(all_runs),
            "seed": 42,
        },
        "best_parameters": best_params,
        "default_parameters": DEFAULT_PARAMS,
        "parameter_bounds": {
            name: {"lower": low, "upper": high}
            for name, (low, high) in PARAM_BOUNDS.items()
        },
        "metrics": metrics,
        "best_run_summary": {
            "start_index": best.start_index,
            "success": best.success,
            "message": best.message,
            "iterations": best.nit,
            "function_evaluations": best.nfev,
        },
        "all_runs": [
            {
                "start_index": item.start_index,
                "objective": item.fun,
                "success": item.success,
                "message": item.message,
                "iterations": item.nit,
                "function_evaluations": item.nfev,
            }
            for item in all_runs
        ],
    }

    best_run.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    calibrator = ModelCalibrator(MODEL_PATH)
    best, all_runs = run_multistart_optimization(calibrator)

    best_params = build_params(best.x)
    best_run = calibrator.run_model(best_params)
    _, metrics = calibrator.evaluate(best.x)

    save_outputs(best=best, all_runs=all_runs, best_run=best_run, metrics=metrics)

    print("Calibration finished.")
    print("Method: scipy.optimize.minimize (L-BFGS-B, multistart)")
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
