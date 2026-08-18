"""Build ICE fixed-distance segments from raw eVED records."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REFERENCE_DATE = datetime(2017, 11, 1)
KG_PER_LB = 0.45359237

RAW_COLUMNS = [
    "DayNum",
    "VehId",
    "Trip",
    "Timestamp(ms)",
    "Vehicle Speed[km/h]",
    "OAT[DegC]",
    "Elevation Smoothed[m]",
    "Gradient",
    "Energy_Consumption",
    "Matchted Latitude[deg]",
    "Matched Longitude[deg]",
    "Speed Limit[km/h]",
]


@dataclass(frozen=True)
class PreprocessConfig:
    segment_m: float = 250.0
    min_segment_m: float = 100.0
    max_segment_m: float = 375.0
    min_segment_duration_s: float = 5.0
    max_segment_duration_s: float = 600.0
    min_target_coverage: float = 0.95
    max_gap_s: float = 5.0
    max_gps_step_m: float = 500.0
    min_distance_consistency: float = 0.5
    max_distance_consistency: float = 1.5
    max_avg_speed_kmh: float = 150.0
    max_energy_kwh_per_100km: float = 200.0


def haversine_step_m(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Distance from each row to the next row; final row is zero."""
    if len(latitude) == 0:
        return np.array([], dtype=float)
    lat1 = np.radians(latitude)
    lon1 = np.radians(longitude)
    lat2 = np.roll(lat1, -1)
    lon2 = np.roll(lon1, -1)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 6_371_008.8 * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1 - a)))
    distance[-1] = 0.0
    return distance


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[mask], weights=weights[mask])) if mask.any() else math.nan


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    if not np.isfinite(mean):
        return math.nan
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.sqrt(np.average(np.square(values[mask] - mean), weights=weights[mask])))


def parse_engine_displacement_l(value: object) -> float:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)L", str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else math.nan


def load_ice_static(raw_dir: Path) -> pd.DataFrame:
    static = pd.read_excel(raw_dir / "VED_Static_Data_ICE&HEV.xlsx")
    static = static.rename(
        columns={
            "Vehicle Type": "powertrain",
            "Vehicle Class": "vehicle_class",
            "Engine Configuration & Displacement": "engine_configuration",
            "Transmission": "transmission",
            "Drive Wheels": "drive_wheels",
            "Generalized_Weight": "generalized_weight_lb",
        }
    )
    static = static[static["powertrain"].eq("ICE")].copy()
    static["VehId"] = pd.to_numeric(static["VehId"], errors="coerce").astype("Int64")
    static["generalized_weight_lb"] = pd.to_numeric(static["generalized_weight_lb"], errors="coerce")
    static["vehicle_weight_kg"] = static["generalized_weight_lb"] * KG_PER_LB
    static["engine_displacement_l"] = static["engine_configuration"].map(parse_engine_displacement_l)
    return static.drop_duplicates("VehId").set_index("VehId")


def prepare_trip(group: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    trip = group.sort_values("Timestamp(ms)", kind="mergesort").drop_duplicates("Timestamp(ms)").copy()
    timestamp = pd.to_numeric(trip["Timestamp(ms)"], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(timestamp, append=np.nan) / 1000.0
    positive = dt[np.isfinite(dt) & (dt > 0) & (dt <= config.max_gap_s)]
    final_dt = min(1.0, float(np.median(positive))) if len(positive) else 1.0
    dt[-1] = final_dt

    latitude = pd.to_numeric(trip["Matchted Latitude[deg]"], errors="coerce").to_numpy(dtype=float)
    longitude = pd.to_numeric(trip["Matched Longitude[deg]"], errors="coerce").to_numpy(dtype=float)
    distance = haversine_step_m(latitude, longitude)

    valid_time = np.isfinite(dt) & (dt > 0) & (dt <= config.max_gap_s)
    # GPS updates roughly every 3 seconds while other OBD signals update more
    # frequently. Repeated coordinates therefore make raw-row dt unsuitable
    # for an implied GPS-speed check. Reject only clearly impossible spatial
    # jumps here; trip-level distance consistency is checked downstream.
    valid_gps = np.isfinite(distance) & (distance <= config.max_gps_step_m)

    trip["dt_s"] = np.where(valid_time, dt, 0.0)
    trip["step_distance_m"] = np.where(valid_time & valid_gps, distance, 0.0)
    trip["invalid_time_gap"] = ~valid_time
    trip["invalid_gps_step"] = valid_time & ~valid_gps
    trip["cum_distance_m"] = trip["step_distance_m"].cumsum().shift(fill_value=0.0)
    trip["segment_index"] = np.floor(trip["cum_distance_m"] / config.segment_m).astype(int)
    return trip


def profile_trip(trip: pd.DataFrame, source_file: str, config: PreprocessConfig) -> dict:
    duration = float(trip["dt_s"].sum())
    energy = pd.to_numeric(trip["Energy_Consumption"], errors="coerce").to_numpy(dtype=float)
    dt = trip["dt_s"].to_numpy(dtype=float)
    valid_target_duration = float(dt[np.isfinite(energy)].sum())
    speed = pd.to_numeric(trip["Vehicle Speed[km/h]"], errors="coerce").to_numpy(dtype=float)
    speed_distance = float(np.nansum(speed / 3.6 * dt))
    gps_distance = float(trip["step_distance_m"].sum())
    distance_consistency = gps_distance / speed_distance if speed_distance > 0 else math.nan
    eligible = (
        duration >= config.min_segment_duration_s
        and valid_target_duration / duration >= config.min_target_coverage
        and np.isfinite(distance_consistency)
        and config.min_distance_consistency <= distance_consistency <= config.max_distance_consistency
    ) if duration > 0 else False
    return {
        "source_file": source_file,
        "VehId": int(trip["VehId"].iloc[0]),
        "Trip": int(trip["Trip"].iloc[0]),
        "rows": len(trip),
        "duration_s": duration,
        "distance_m": gps_distance,
        "speed_integrated_distance_m": speed_distance,
        "distance_consistency_ratio": distance_consistency,
        "target_coverage": valid_target_duration / duration if duration > 0 else 0.0,
        "invalid_time_gaps": int(trip["invalid_time_gap"].sum()),
        "invalid_gps_steps": int(trip["invalid_gps_step"].sum()),
        "eligible": bool(eligible),
    }


def segment_trip(
    trip: pd.DataFrame,
    static_row: pd.Series,
    source_file: str,
    config: PreprocessConfig,
) -> list[dict]:
    records: list[dict] = []
    for segment_index, segment in trip.groupby("segment_index", sort=True):
        dt = segment["dt_s"].to_numpy(dtype=float)
        duration = float(dt.sum())
        distance = float(segment["step_distance_m"].sum())
        energy = pd.to_numeric(segment["Energy_Consumption"], errors="coerce").to_numpy(dtype=float)
        target_mask = np.isfinite(energy) & (dt > 0)
        target_duration = float(dt[target_mask].sum())
        target_coverage = target_duration / duration if duration > 0 else 0.0

        speed = pd.to_numeric(segment["Vehicle Speed[km/h]"], errors="coerce").to_numpy(dtype=float)
        speed_integrated_distance = float(np.nansum(speed / 3.6 * dt))
        distance_consistency = distance / speed_integrated_distance if speed_integrated_distance > 0 else math.nan

        if (
            distance < config.min_segment_m
            or distance > config.max_segment_m
            or duration < config.min_segment_duration_s
            or duration > config.max_segment_duration_s
            or target_coverage < config.min_target_coverage
            or not np.isfinite(distance_consistency)
            or not config.min_distance_consistency <= distance_consistency <= config.max_distance_consistency
        ):
            continue

        speed_limit = pd.to_numeric(segment["Speed Limit[km/h]"], errors="coerce").to_numpy(dtype=float)
        gradient = pd.to_numeric(segment["Gradient"], errors="coerce").to_numpy(dtype=float)
        elevation = pd.to_numeric(segment["Elevation Smoothed[m]"], errors="coerce").to_numpy(dtype=float)
        temperature = pd.to_numeric(segment["OAT[DegC]"], errors="coerce").to_numpy(dtype=float)
        day_num = float(pd.to_numeric(segment["DayNum"], errors="coerce").median())
        observed_at = REFERENCE_DATE + timedelta(days=day_num - 1)
        median_limit = float(np.nanmedian(speed_limit)) if np.isfinite(speed_limit).any() else math.nan
        average_speed = weighted_mean(speed, dt)
        segment_energy = float(np.sum(energy[target_mask] * dt[target_mask]))
        energy_per_100km = segment_energy / distance * 100_000
        if (
            not np.isfinite(average_speed)
            or average_speed > config.max_avg_speed_kmh
            or segment_energy <= 0
            or energy_per_100km > config.max_energy_kwh_per_100km
        ):
            continue
        elevation_delta = np.diff(elevation)
        elevation_gain = float(np.nansum(np.where(elevation_delta > 0, elevation_delta, 0.0)))

        vehicle_id = int(segment["VehId"].iloc[0])
        trip_id = int(segment["Trip"].iloc[0])
        records.append(
            {
                "segment_id": f"{Path(source_file).stem}_v{vehicle_id}_t{trip_id}_s{int(segment_index)}",
                "source_file": source_file,
                "vehicle_id": vehicle_id,
                "trip_id": trip_id,
                "segment_index": int(segment_index),
                "powertrain": "ICE",
                "vehicle_weight_kg": float(static_row.get("vehicle_weight_kg", math.nan)),
                "engine_displacement_l": float(static_row.get("engine_displacement_l", math.nan)),
                "transmission": static_row.get("transmission"),
                "drive_wheels": static_row.get("drive_wheels"),
                "distance_m": distance,
                "speed_integrated_distance_m": speed_integrated_distance,
                "distance_consistency_ratio": distance_consistency,
                "travel_time_s": duration,
                "avg_speed_kmh": average_speed,
                "speed_std_kmh": weighted_std(speed, dt),
                "speed_limit_kmh": median_limit,
                "congestion_ratio": average_speed / median_limit if median_limit > 0 else math.nan,
                "stop_ratio": float(dt[(speed < 5) & np.isfinite(speed)].sum() / duration),
                "low_speed_ratio": float(
                    dt[(speed < 0.5 * speed_limit) & np.isfinite(speed) & np.isfinite(speed_limit)].sum() / duration
                ),
                "avg_gradient": weighted_mean(gradient, dt),
                "max_gradient": float(np.nanmax(gradient)) if np.isfinite(gradient).any() else math.nan,
                "min_gradient": float(np.nanmin(gradient)) if np.isfinite(gradient).any() else math.nan,
                "elevation_gain_m": elevation_gain,
                "temperature_c": float(np.nanmedian(temperature)) if np.isfinite(temperature).any() else math.nan,
                "hour": observed_at.hour,
                "weekday": observed_at.weekday(),
                "target_coverage": target_coverage,
                "segment_energy_kwh": segment_energy,
            }
        )
    return records


def resolve_files(raw_dir: Path, limit_files: int | None) -> list[Path]:
    files = sorted((raw_dir / "eVED" / "eVED").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No eVED CSV files found under {raw_dir}")
    return files[:limit_files] if limit_files else files


def combine_csv_parts(parts: Iterable[Path], output_path: Path) -> int:
    """Combine deterministic per-week CSV parts into one model-ready CSV."""
    frames = [pd.read_csv(path, low_memory=False) for path in sorted(parts)]
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(combined)


def process_files(
    root: Path,
    config: PreprocessConfig,
    mode: str,
    limit_files: int | None,
    overwrite: bool,
) -> None:
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    profile_dir = processed_dir / "trip_profiles"
    segment_dir = processed_dir / f"segments_{int(config.segment_m)}m"
    files = resolve_files(raw_dir, limit_files)
    static = load_ice_static(raw_dir)

    for directory in [profile_dir, segment_dir]:
        if overwrite and directory.exists() and directory.resolve().is_relative_to(processed_dir.resolve()):
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "input_files": len(files),
        "config": asdict(config),
        "powertrain": "ICE",
        "target": "segment_energy_kwh",
    }
    (processed_dir / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {"files": 0, "ice_rows": 0, "trips": 0, "eligible_trips": 0, "segments": 0}
    for index, path in enumerate(files, 1):
        print(f"[{index:02d}/{len(files)}] {path.name}", flush=True)
        frame = pd.read_csv(path, usecols=RAW_COLUMNS, low_memory=False)
        frame["VehId"] = pd.to_numeric(frame["VehId"], errors="coerce").astype("Int64")
        frame["Trip"] = pd.to_numeric(frame["Trip"], errors="coerce").astype("Int64")
        frame = frame[frame["VehId"].isin(static.index)].dropna(subset=["VehId", "Trip"])
        summary["files"] += 1
        summary["ice_rows"] += len(frame)

        profiles: list[dict] = []
        segments: list[dict] = []
        for (vehicle_id, _), group in frame.groupby(["VehId", "Trip"], sort=False):
            trip = prepare_trip(group, config)
            profile = profile_trip(trip, path.name, config)
            profiles.append(profile)
            summary["trips"] += 1
            if profile["eligible"]:
                summary["eligible_trips"] += 1
                if mode in {"build", "all"}:
                    segments.extend(segment_trip(trip, static.loc[int(vehicle_id)], path.name, config))

        if mode in {"profile", "all"}:
            pd.DataFrame(profiles).to_csv(profile_dir / f"{path.stem}_trips.csv", index=False, encoding="utf-8-sig")
        if mode in {"build", "all"}:
            pd.DataFrame(segments).to_csv(segment_dir / f"{path.stem}_segments.csv", index=False, encoding="utf-8-sig")
            summary["segments"] += len(segments)

    if mode in {"profile", "all"}:
        summary["master_trip_profiles"] = combine_csv_parts(
            profile_dir.glob("eVED_*_trips.csv"), profile_dir / "ice_trip_profiles.csv"
        )
    if mode in {"build", "all"}:
        summary["master_segments"] = combine_csv_parts(
            segment_dir.glob("eVED_*_segments.csv"), segment_dir / "ice_segments_250m.csv"
        )

    (processed_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess raw eVED records for EcoRoute")
    parser.add_argument("--mode", choices=["profile", "build", "all"], default="all")
    parser.add_argument("--segment-m", type=float, default=250.0)
    parser.add_argument("--min-segment-m", type=float, default=100.0)
    parser.add_argument("--min-target-coverage", type=float, default=0.95)
    parser.add_argument("--max-gap-s", type=float, default=5.0)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[2]
    config = PreprocessConfig(
        segment_m=args.segment_m,
        min_segment_m=args.min_segment_m,
        min_target_coverage=args.min_target_coverage,
        max_gap_s=args.max_gap_s,
    )
    process_files(root, config, args.mode, args.limit_files, args.overwrite)
