"""PyTorch MLP training for EcoRoute segment energy prediction."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .training import FEATURES, RANDOM_STATE, TARGET, regression_metrics


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HIDDEN_DIMS = [128, 128, 64, 32]
DROPOUTS = [0.10, 0.10, 0.05]
MODEL_NAME = "dnn_mlp"


class EnergyMLP(nn.Module):
    """Four-hidden-layer MLP for standardized tabular inputs."""

    def __init__(self, input_dim: int = len(FEATURES)) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def set_reproducible_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def make_loader(
    features: np.ndarray,
    targets: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    feature_tensor = torch.from_numpy(features)
    if targets is None:
        dataset = TensorDataset(feature_tensor)
    else:
        dataset = TensorDataset(feature_tensor, torch.from_numpy(targets))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=bool(shuffle and len(dataset) % batch_size == 1),
    )


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    target_scale: float,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    absolute_error_sum = 0.0
    row_count = 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(features)
        loss = criterion(predictions, targets)
        batch_rows = len(targets)
        loss_sum += float(loss.item()) * batch_rows
        absolute_error_sum += float(torch.abs(predictions - targets).sum().item()) * target_scale
        row_count += batch_rows
    return loss_sum / row_count, absolute_error_sum / row_count


@torch.no_grad()
def predict(
    model: nn.Module,
    features: np.ndarray,
    target_mean: float,
    target_scale: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = make_loader(features, None, batch_size, False, device.type == "cuda")
    batches: list[np.ndarray] = []
    for (batch_features,) in loader:
        standardized = model(batch_features.to(device, non_blocking=True))
        restored = standardized * target_scale + target_mean
        batches.append(restored.detach().cpu().numpy())
    return np.clip(np.concatenate(batches), 0.0, None)


def save_loss_curve(history: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Huber loss")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation Huber loss")
    axes[0].axvline(history.loc[history["is_best"], "epoch"].iloc[-1], color="#16A34A", linestyle="--")
    axes[0].set_title("Standardized-target loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["train_mae_kwh"], label="Train MAE")
    axes[1].plot(history["epoch"], history["validation_mae_kwh"], label="Validation MAE")
    axes[1].axvline(history.loc[history["is_best"], "epoch"].iloc[-1], color="#16A34A", linestyle="--")
    axes[1].set_title("Energy prediction error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (kWh)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("EcoRoute DNN training history", fontsize=15)
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_actual_vs_predicted(predictions: pd.DataFrame, figures_dir: Path) -> None:
    test = predictions.loc[predictions["split"].eq("test")]
    visible = test.sample(min(20_000, len(test)), random_state=RANDOM_STATE)
    upper = float(
        np.quantile(
            np.concatenate([visible["actual_energy_kwh"], visible["predicted_energy_kwh"]]),
            0.995,
        )
    )
    upper = max(upper, 1e-6)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        visible["actual_energy_kwh"],
        visible["predicted_energy_kwh"],
        s=8,
        alpha=0.2,
        color="#7C3AED",
        edgecolors="none",
    )
    ax.plot([0, upper], [0, upper], color="#DC2626", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_title("DNN actual vs predicted energy (test split)")
    ax.set_xlabel("Actual segment energy (kWh)")
    ax.set_ylabel("Predicted segment energy (kWh)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_error_distribution(predictions: pd.DataFrame, figures_dir: Path) -> None:
    test_error = predictions.loc[predictions["split"].eq("test"), "error_kwh"]
    lower, upper = test_error.quantile([0.01, 0.99])
    visible = test_error.loc[test_error.between(lower, upper)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(visible, bins=80, density=True, color="#8B5CF6", alpha=0.75)
    ax.axvline(0, color="#111827", linestyle="--", linewidth=1.2)
    ax.set_title("DNN prediction error (test split, 1st-99th percentile view)")
    ax.set_xlabel("Prediction - actual (kWh)")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "error_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_baseline_comparison(
    dnn_metrics: pd.DataFrame,
    baseline_metrics_path: Path,
    figures_dir: Path,
) -> None:
    baseline = pd.read_csv(baseline_metrics_path)
    comparison = pd.concat(
        [baseline.loc[baseline["split"].eq("test")], dnn_metrics.loc[dnn_metrics["split"].eq("test")]],
        ignore_index=True,
    )
    specs = [
        ("mae_kwh", "MAE (kWh)", "lower is better"),
        ("rmse_kwh", "RMSE (kWh)", "lower is better"),
        ("r2", "R-squared", "higher is better"),
        ("smape_pct", "sMAPE (%)", "lower is better"),
    ]
    colors = ["#9CA3AF", "#60A5FA", "#22C55E", "#8B5CF6"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (column, title, subtitle) in zip(axes.flat, specs, strict=True):
        values = comparison[column]
        bars = ax.bar(comparison["model"], values, color=colors[: len(comparison)])
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
    fig.suptitle("Baseline vs DNN (same test split)", fontsize=15)
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_vs_dnn.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_dnn(
    root: Path,
    max_epochs: int = 150,
    batch_size: int = 512,
    patience: int = 15,
    learning_rate: float = 1e-3,
    max_rows: int | None = None,
    requested_device: str = "auto",
) -> None:
    set_reproducible_seed()
    device = resolve_device(requested_device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data_path = root / "data" / "processed" / "segments_250m" / "ice_segments_250m.csv"
    split_path = root / "results" / "baseline" / "split_assignments.csv"
    baseline_metrics_path = root / "results" / "baseline" / "metrics.csv"
    model_dir = root / "models" / "dnn"
    result_dir = root / "results" / "dnn"
    figures_dir = result_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not split_path.exists() or not baseline_metrics_path.exists():
        raise FileNotFoundError("Run train.py first so the shared baseline split and metrics exist")

    frame = pd.read_csv(data_path, low_memory=False)
    assignment = pd.read_csv(
        split_path,
        usecols=["segment_id", "split", "trip_group_id"],
        dtype={"split": "string", "trip_group_id": "string"},
    )
    frame = frame.merge(assignment, on="segment_id", how="left", validate="one_to_one")
    if frame["split"].isna().any():
        raise ValueError("Some processed rows have no saved baseline split assignment")
    if max_rows and max_rows < len(frame):
        frame = frame.sample(max_rows, random_state=RANDOM_STATE).sort_index().reset_index(drop=True)

    required = set(FEATURES + [TARGET, "segment_id", "vehicle_id", "trip_id", "split"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if (frame[TARGET] <= 0).any():
        raise ValueError("Target must be strictly positive")

    split = frame["split"]
    train_mask = split.eq("train").to_numpy()
    validation_mask = split.eq("validation").to_numpy()
    test_mask = split.eq("test").to_numpy()
    if min(train_mask.sum(), validation_mask.sum(), test_mask.sum()) == 0:
        raise ValueError("Train, validation, and test splits must all contain rows")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed_train = imputer.fit_transform(frame.loc[train_mask, FEATURES])
    scaler.fit(imputed_train)
    all_features = scaler.transform(imputer.transform(frame[FEATURES])).astype(np.float32)

    target_values = frame[TARGET].to_numpy(dtype=np.float32)
    target_mean = float(target_values[train_mask].mean())
    target_scale = float(target_values[train_mask].std())
    if target_scale <= 0:
        raise ValueError("Training target has zero variance")
    standardized_targets = ((target_values - target_mean) / target_scale).astype(np.float32)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        all_features[train_mask], standardized_targets[train_mask], batch_size, True, pin_memory
    )
    validation_loader = make_loader(
        all_features[validation_mask],
        standardized_targets[validation_mask],
        batch_size,
        False,
        pin_memory,
    )

    model = EnergyMLP().to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    print(
        f"Training {MODEL_NAME} on {device} | rows={len(frame):,} | "
        f"parameters={count_parameters(model):,}",
        flush=True,
    )
    history_rows: list[dict[str, float | int | bool]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_mae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_absolute_error_sum = 0.0
        train_rows = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            batch_predictions = model(batch_features)
            loss = criterion(batch_predictions, batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            rows = len(batch_targets)
            train_loss_sum += float(loss.item()) * rows
            train_absolute_error_sum += (
                float(torch.abs(batch_predictions.detach() - batch_targets).sum().item()) * target_scale
            )
            train_rows += rows

        train_loss = train_loss_sum / train_rows
        train_mae = train_absolute_error_sum / train_rows
        validation_loss, validation_mae = evaluate_loader(
            model, validation_loader, criterion, target_scale, device
        )
        scheduler.step(validation_loss)
        improved = validation_mae < best_validation_mae - 1e-6
        if improved:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_mae_kwh": train_mae,
                "validation_mae_kwh": validation_mae,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )
        print(
            f"epoch={epoch:03d} train_mae={train_mae:.6f} "
            f"val_mae={validation_mae:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("DNN training completed without a best checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    metrics_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split_name, mask in [("validation", validation_mask), ("test", test_mask)]:
        split_predictions = predict(
            model, all_features[mask], target_mean, target_scale, batch_size, device
        )
        actual = target_values[mask]
        row: dict[str, float | int | str] = {
            "model": MODEL_NAME,
            "split": split_name,
            "rows": int(mask.sum()),
        }
        row.update(regression_metrics(pd.Series(actual), split_predictions))
        metrics_rows.append(row)
        print(row, flush=True)

        prediction_frame = frame.loc[
            mask, ["segment_id", "vehicle_id", "trip_id", "trip_group_id", "split"]
        ].copy()
        prediction_frame.insert(0, "model", MODEL_NAME)
        prediction_frame["actual_energy_kwh"] = actual
        prediction_frame["predicted_energy_kwh"] = split_predictions
        prediction_frame["error_kwh"] = split_predictions - actual
        prediction_frame["absolute_error_kwh"] = np.abs(prediction_frame["error_kwh"])
        prediction_frames.append(prediction_frame)

    history = pd.DataFrame(history_rows)
    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    history.to_csv(result_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(result_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(result_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    checkpoint = {
        "format_version": 1,
        "model_name": MODEL_NAME,
        "model_state_dict": best_state,
        "input_features": FEATURES,
        "hidden_dims": HIDDEN_DIMS,
        "dropouts": DROPOUTS,
        "imputer_statistics": torch.tensor(imputer.statistics_, dtype=torch.float32),
        "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
        "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "best_epoch": best_epoch,
        "random_state": RANDOM_STATE,
    }
    torch.save(checkpoint, model_dir / "best_model.pt")

    save_loss_curve(history, figures_dir)
    save_actual_vs_predicted(predictions, figures_dir)
    save_error_distribution(predictions, figures_dir)
    save_baseline_comparison(metrics, baseline_metrics_path, figures_dir)

    split_summary = (
        frame.groupby("split", observed=True)
        .agg(rows=("segment_id", "size"), vehicles=("vehicle_id", "nunique"), trips=("trip_group_id", "nunique"))
        .reset_index()
    )
    split_summary.to_csv(result_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    config = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_target": "3.14",
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "random_state": RANDOM_STATE,
        "target": TARGET,
        "features": FEATURES,
        "architecture": [len(FEATURES), *HIDDEN_DIMS, 1],
        "activation": "SiLU",
        "normalization": "BatchNorm1d",
        "dropouts": DROPOUTS,
        "trainable_parameters": count_parameters(model),
        "loss": "SmoothL1Loss(beta=1.0) on standardized target",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "best_epoch": best_epoch,
        "best_validation_mae_kwh": best_validation_mae,
        "rows": len(frame),
        "shared_split_source": str(split_path.relative_to(root)),
        "test_usage": "Final evaluation only; not used for optimization or early stopping",
        "model_directory": str(model_dir.relative_to(root)),
        "result_directory": str(result_dir.relative_to(root)),
    }
    (result_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Best epoch: {best_epoch} | validation MAE: {best_validation_mae:.6f} kWh", flush=True)
    print(f"Model saved to: {model_dir / 'best_model.pt'}", flush=True)
    print(f"Results saved to: {result_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the EcoRoute PyTorch DNN")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-rows", type=int, help="Optional deterministic smoke-test sample")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    train_dnn(
        root=root,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        max_rows=args.max_rows,
        requested_device=args.device,
    )
