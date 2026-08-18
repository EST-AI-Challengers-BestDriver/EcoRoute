"""Leakage-safe baseline training for EcoRoute segment energy prediction."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# Keep scikit-learn deterministic and compatible with restricted Windows
# environments where worker pipes/threads may be unavailable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RANDOM_STATE = 20260811
TARGET = "segment_energy_kwh"
FEATURES = [
    "vehicle_weight_kg",
    "engine_displacement_l",
    "distance_m",
    "travel_time_s",
    "avg_speed_kmh",
    "speed_std_kmh",
    "speed_limit_kmh",
    "congestion_ratio",
    "stop_ratio",
    "low_speed_ratio",
    "avg_gradient",
    "max_gradient",
    "min_gradient",
    "elevation_gain_m",
    "hour",
    "weekday",
]


def split_by_trip(frame: pd.DataFrame) -> pd.Series:
    """Return deterministic 70/15/15 split labels without Trip leakage."""
    groups = frame["vehicle_id"].astype(str) + "_" + frame["trip_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(splitter.split(frame, groups=groups))
    temp = frame.iloc[temp_idx]
    temp_groups = groups.iloc[temp_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE + 1)
    val_rel, test_rel = next(second.split(temp, groups=temp_groups))

    labels = pd.Series(index=frame.index, dtype="string")
    labels.iloc[train_idx] = "train"
    labels.iloc[temp_idx[val_rel]] = "validation"
    labels.iloc[temp_idx[test_rel]] = "test"
    if labels.isna().any():
        raise RuntimeError("Split assignment left unlabeled rows")
    return labels


def build_models() -> dict[str, object]:
    linear = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    hist_gb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=300,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=0.1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {
        "dummy_median": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("regressor", DummyRegressor(strategy="median"))]
        ),
        "linear_regression": linear,
        "hist_gradient_boosting": hist_gb,
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    denominator = (np.abs(y_true.to_numpy()) + np.abs(y_pred)) / 2
    smape = np.mean(np.divide(np.abs(y_true.to_numpy() - y_pred), denominator, out=np.zeros_like(y_pred), where=denominator > 0))
    return {
        "mae_kwh": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kwh": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape_pct": float(smape * 100),
    }


def save_model_metrics_figure(metrics: pd.DataFrame, figures_dir: Path) -> None:
    """Save a compact comparison of the test metrics for every baseline."""
    test = metrics.loc[metrics["split"].eq("test")].set_index("model")
    specs = [
        ("mae_kwh", "MAE (kWh)", "lower is better"),
        ("rmse_kwh", "RMSE (kWh)", "lower is better"),
        ("r2", "R-squared", "higher is better"),
        ("smape_pct", "sMAPE (%)", "lower is better"),
    ]
    colors = ["#9CA3AF", "#60A5FA", "#22C55E"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (column, title, subtitle) in zip(axes.flat, specs, strict=True):
        values = test[column]
        bars = ax.bar(values.index, values.values, color=colors[: len(values)])
        ax.axhline(0, color="#374151", linewidth=0.8)
        ax.set_title(f"{title}\n{subtitle}")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            y_offset = 3 if value >= 0 else -14
            ax.annotate(
                f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
    fig.suptitle("EcoRoute baseline model comparison (test split)", fontsize=15)
    fig.tight_layout()
    fig.savefig(figures_dir / "model_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_actual_vs_predicted_figure(predictions: pd.DataFrame, figures_dir: Path) -> None:
    """Save test-set actual-versus-predicted scatter plots."""
    test = predictions.loc[predictions["split"].eq("test")]
    model_names = list(test["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5), squeeze=False)
    for ax, name in zip(axes.flat, model_names, strict=True):
        values = test.loc[test["model"].eq(name)]
        if len(values) > 15_000:
            values = values.sample(15_000, random_state=RANDOM_STATE)
        upper = float(
            np.quantile(
                np.concatenate([values["actual_energy_kwh"], values["predicted_energy_kwh"]]),
                0.995,
            )
        )
        upper = max(upper, 1e-6)
        ax.scatter(
            values["actual_energy_kwh"],
            values["predicted_energy_kwh"],
            s=7,
            alpha=0.18,
            color="#2563EB",
            edgecolors="none",
        )
        ax.plot([0, upper], [0, upper], linestyle="--", color="#DC2626", linewidth=1.2)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_title(name)
        ax.set_xlabel("Actual segment energy (kWh)")
        ax.set_ylabel("Predicted segment energy (kWh)")
        ax.grid(alpha=0.2)
    fig.suptitle("Actual vs predicted energy (test split)", fontsize=15)
    fig.tight_layout()
    fig.savefig(figures_dir / "actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_error_distribution_figure(predictions: pd.DataFrame, figures_dir: Path) -> None:
    """Save comparable test-error distributions with robust display limits."""
    test = predictions.loc[predictions["split"].eq("test")]
    lower, upper = test["error_kwh"].quantile([0.01, 0.99])
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in test.groupby("model", sort=False):
        visible = values.loc[values["error_kwh"].between(lower, upper), "error_kwh"]
        ax.hist(visible, bins=80, density=True, alpha=0.35, label=name)
    ax.axvline(0, color="#111827", linestyle="--", linewidth=1.2)
    ax.set_title("Prediction error distribution (test split, 1st-99th percentile view)")
    ax.set_xlabel("Prediction - actual (kWh)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "error_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def calculate_learning_curve(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    split: pd.Series,
    trip_group_id: pd.Series,
) -> pd.DataFrame:
    """Measure HistGradientBoosting MAE as progressively more training trips are used."""
    training_groups = trip_group_id.loc[split.eq("train")].drop_duplicates().to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(training_groups)
    validation_mask = split.eq("validation")
    rows: list[dict[str, float | int]] = []
    for fraction in [0.10, 0.25, 0.50, 0.75, 1.00]:
        group_count = max(1, int(np.ceil(len(training_groups) * fraction)))
        selected_groups = set(training_groups[:group_count])
        training_mask = split.eq("train") & trip_group_id.isin(selected_groups)
        curve_model = clone(model)
        curve_model.fit(X.loc[training_mask], y.loc[training_mask])
        train_prediction = np.clip(curve_model.predict(X.loc[training_mask]), 0.0, None)
        validation_prediction = np.clip(curve_model.predict(X.loc[validation_mask]), 0.0, None)
        rows.append(
            {
                "training_fraction": fraction,
                "training_trips": group_count,
                "training_rows": int(training_mask.sum()),
                "train_mae_kwh": float(mean_absolute_error(y.loc[training_mask], train_prediction)),
                "validation_mae_kwh": float(
                    mean_absolute_error(y.loc[validation_mask], validation_prediction)
                ),
            }
        )
    return pd.DataFrame(rows)


def save_learning_curve_figure(curve: pd.DataFrame, figures_dir: Path) -> None:
    """Save the HistGradientBoosting data-size learning curve."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(curve["training_rows"], curve["train_mae_kwh"], marker="o", label="Train MAE")
    ax.plot(
        curve["training_rows"],
        curve["validation_mae_kwh"],
        marker="o",
        label="Validation MAE",
    )
    ax.set_title("HistGradientBoosting learning curve")
    ax.set_xlabel("Training rows (trip-group subsets)")
    ax.set_ylabel("MAE (kWh)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "learning_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_baselines(root: Path, max_rows: int | None = None) -> None:
    data_path = root / "data" / "processed" / "segments_250m" / "ice_segments_250m.csv"
    model_dir = root / "models" / "baseline"
    result_dir = root / "results" / "baseline"
    figures_dir = result_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path, low_memory=False)
    if max_rows and max_rows < len(frame):
        frame = frame.sample(max_rows, random_state=RANDOM_STATE).sort_index().reset_index(drop=True)

    required = set(FEATURES + [TARGET, "segment_id", "vehicle_id", "trip_id"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if (frame[TARGET] <= 0).any():
        raise ValueError("Target must be strictly positive after preprocessing")

    split = split_by_trip(frame)
    assignment = frame[["segment_id", "vehicle_id", "trip_id"]].copy()
    assignment["split"] = split
    assignment["trip_group_id"] = assignment["vehicle_id"].astype(str) + "_" + assignment["trip_id"].astype(str)
    assignment.to_csv(result_dir / "split_assignments.csv", index=False, encoding="utf-8-sig")

    split_summary = (
        assignment.groupby("split")
        .agg(
            rows=("segment_id", "size"),
            vehicles=("vehicle_id", "nunique"),
            trips=("trip_group_id", "nunique"),
        )
        .reset_index()
    )
    split_summary.to_csv(result_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    X = frame[FEATURES]
    y = frame[TARGET]
    metrics: list[dict] = []
    predictions: list[pd.DataFrame] = []
    models = build_models()
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        train_mask = split.eq("train")
        model.fit(X.loc[train_mask], y.loc[train_mask])
        joblib.dump(model, model_dir / f"{name}.joblib")
        for split_name in ["validation", "test"]:
            mask = split.eq(split_name)
            prediction = np.clip(model.predict(X.loc[mask]), 0.0, None)
            row = {"model": name, "split": split_name, "rows": int(mask.sum())}
            row.update(regression_metrics(y.loc[mask], prediction))
            metrics.append(row)
            prediction_frame = assignment.loc[
                mask, ["segment_id", "vehicle_id", "trip_id", "trip_group_id", "split"]
            ].copy()
            prediction_frame.insert(0, "model", name)
            prediction_frame["actual_energy_kwh"] = y.loc[mask].to_numpy()
            prediction_frame["predicted_energy_kwh"] = prediction
            prediction_frame["error_kwh"] = prediction - y.loc[mask].to_numpy()
            prediction_frame["absolute_error_kwh"] = np.abs(prediction_frame["error_kwh"])
            predictions.append(prediction_frame)
            print(row, flush=True)

    metrics_frame = pd.DataFrame(metrics)
    predictions_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame.to_csv(result_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(result_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    print("Calculating HistGradientBoosting learning curve...", flush=True)
    learning_curve = calculate_learning_curve(
        models["hist_gradient_boosting"], X, y, split, assignment["trip_group_id"]
    )
    learning_curve.to_csv(result_dir / "learning_curve.csv", index=False, encoding="utf-8-sig")

    save_model_metrics_figure(metrics_frame, figures_dir)
    save_actual_vs_predicted_figure(predictions_frame, figures_dir)
    save_error_distribution_figure(predictions_frame, figures_dir)
    save_learning_curve_figure(learning_curve, figures_dir)
    config = {
        "python_target": "3.14",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "target": TARGET,
        "features": FEATURES,
        "excluded_first_baseline": {
            "temperature_c": "67.8% missing",
            "transmission": "not reliably available for service input",
            "drive_wheels": "not reliably available for service input",
            "identifiers_and_quality_columns": "tracking/QC only",
        },
        "rows": len(frame),
        "model_directory": str(model_dir.relative_to(root)),
        "result_directory": str(result_dir.relative_to(root)),
    }
    (result_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Models saved to: {model_dir}", flush=True)
    print(f"Results saved to: {result_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EcoRoute baseline regressors")
    parser.add_argument("--max-rows", type=int, help="Optional deterministic smoke-test sample")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    train_baselines(root, args.max_rows)
