"""EcoRoute phase-1 data audit.

Scans every eVED CSV in chunks, joins vehicle static metadata, and writes
machine-readable summaries without loading the full 5+ GB dataset at once.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CORE_COLUMNS = [
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


def clean_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def load_static(root: Path) -> pd.DataFrame:
    raw_dir = root / "data" / "raw"
    ice = pd.read_excel(raw_dir / "VED_Static_Data_ICE&HEV.xlsx")
    plug = pd.read_excel(raw_dir / "VED_Static_Data_PHEV&EV.xlsx")
    ice = ice.rename(columns={"Vehicle Type": "powertrain"})
    plug = plug.rename(columns={"EngineType": "powertrain"})
    static = pd.concat([ice, plug], ignore_index=True)
    static = static.rename(
        columns={
            "Vehicle Class": "vehicle_class",
            "Engine Configuration & Displacement": "engine_configuration",
            "Transmission": "transmission",
            "Drive Wheels": "drive_wheels",
            "Generalized_Weight": "generalized_weight",
        }
    )
    static["VehId"] = pd.to_numeric(static["VehId"], errors="coerce").astype("Int64")
    return static


def describe_series(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    qs = np.quantile(values, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p25": float(qs[3]),
        "p50": float(qs[4]),
        "p75": float(qs[5]),
        "p95": float(qs[6]),
        "p99": float(qs[7]),
        "max": float(qs[8]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--sample-size", type=int, default=300_000)
    args = parser.parse_args()

    root = args.root.resolve()
    source_dir = root / "data" / "raw" / "eVED" / "eVED"
    output_dir = root / "data" / "processed" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No eVED CSV files found in {source_dir}")

    static = load_static(root)
    static_by_vehicle = static.drop_duplicates("VehId").set_index("VehId")
    static_duplicate_ids = sorted(
        int(v) for v in static.loc[static.duplicated("VehId", keep=False), "VehId"].dropna().unique()
    )

    headers = {p.name: list(pd.read_csv(p, nrows=0).columns) for p in files}
    canonical = headers[files[0].name]
    schema_mismatches = {name: cols for name, cols in headers.items() if cols != canonical}

    total_rows = 0
    nonnull = Counter()
    vehicle_rows = Counter()
    vehicle_target_rows = Counter()
    vehicle_trips: dict[int, set[int]] = defaultdict(set)
    vehicle_days: dict[int, set[float]] = defaultdict(set)
    file_rows = Counter()
    file_target_rows = Counter()
    energy_sign = Counter()
    core_min = {}
    core_max = {}
    rng = np.random.default_rng(20260811)
    samples: dict[str, list[np.ndarray]] = {c: [] for c in CORE_COLUMNS if c not in {"VehId", "Trip"}}
    sample_seen = Counter()

    started = time.time()
    for file_index, path in enumerate(files, start=1):
        print(f"[{file_index:02d}/{len(files)}] {path.name}", flush=True)
        for chunk in pd.read_csv(path, chunksize=args.chunk_size, low_memory=False):
            n = len(chunk)
            total_rows += n
            file_rows[path.name] += n
            nonnull.update({c: int(v) for c, v in chunk.notna().sum().items()})

            veh = pd.to_numeric(chunk["VehId"], errors="coerce").astype("Int64")
            trip = pd.to_numeric(chunk["Trip"], errors="coerce").astype("Int64")
            vehicle_rows.update({int(k): int(v) for k, v in veh.value_counts().items()})
            valid_pairs = pd.DataFrame({"VehId": veh, "Trip": trip, "DayNum": chunk["DayNum"]}).dropna(
                subset=["VehId"]
            )
            for vehicle_id, group in valid_pairs.groupby("VehId", sort=False):
                vid = int(vehicle_id)
                vehicle_trips[vid].update(int(x) for x in group["Trip"].dropna().unique())
                vehicle_days[vid].update(float(x) for x in group["DayNum"].dropna().unique())

            energy = pd.to_numeric(chunk["Energy_Consumption"], errors="coerce")
            valid_energy = energy.notna()
            file_target_rows[path.name] += int(valid_energy.sum())
            target_vehicles = veh[valid_energy].value_counts()
            vehicle_target_rows.update({int(k): int(v) for k, v in target_vehicles.items()})
            energy_sign["negative"] += int((energy < 0).sum())
            energy_sign["zero"] += int((energy == 0).sum())
            energy_sign["positive"] += int((energy > 0).sum())

            for column in CORE_COLUMNS:
                if column not in chunk or column in {"VehId", "Trip"}:
                    continue
                vals = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=float)
                finite = vals[np.isfinite(vals)]
                if not len(finite):
                    continue
                cmin, cmax = float(finite.min()), float(finite.max())
                core_min[column] = min(core_min.get(column, cmin), cmin)
                core_max[column] = max(core_max.get(column, cmax), cmax)

                # Deterministic bounded random sample for approximate quantiles.
                remaining = max(0, args.sample_size - sample_seen[column])
                if remaining:
                    take = min(remaining, len(finite))
                    idx = rng.choice(len(finite), size=take, replace=False)
                    samples[column].append(finite[idx])
                    sample_seen[column] += take

    elapsed = time.time() - started
    all_vehicle_ids = sorted(vehicle_rows)
    matched_ids = [v for v in all_vehicle_ids if v in static_by_vehicle.index]
    unmatched_ids = [v for v in all_vehicle_ids if v not in static_by_vehicle.index]

    vehicle_records = []
    for vehicle_id in all_vehicle_ids:
        meta = static_by_vehicle.loc[vehicle_id].to_dict() if vehicle_id in static_by_vehicle.index else {}
        row_count = vehicle_rows[vehicle_id]
        target_count = vehicle_target_rows[vehicle_id]
        vehicle_records.append(
            {
                "VehId": vehicle_id,
                "powertrain": clean_scalar(meta.get("powertrain")),
                "vehicle_class": clean_scalar(meta.get("vehicle_class")),
                "generalized_weight": clean_scalar(meta.get("generalized_weight")),
                "rows": row_count,
                "target_rows": target_count,
                "target_rate": target_count / row_count if row_count else 0,
                "trip_count": len(vehicle_trips[vehicle_id]),
                "day_value_count": len(vehicle_days[vehicle_id]),
                "static_matched": vehicle_id in static_by_vehicle.index,
            }
        )

    vehicle_df = pd.DataFrame(vehicle_records)
    grouped_df = (
        vehicle_df.assign(
            powertrain=vehicle_df["powertrain"].fillna("UNMATCHED"),
            vehicle_class=vehicle_df["vehicle_class"].fillna("UNMATCHED"),
        )
        .groupby(["powertrain", "vehicle_class"], dropna=False)
        .agg(
            vehicles=("VehId", "nunique"),
            rows=("rows", "sum"),
            target_rows=("target_rows", "sum"),
            trips=("trip_count", "sum"),
        )
        .reset_index()
    )
    grouped_df["target_rate"] = grouped_df["target_rows"] / grouped_df["rows"]

    file_df = pd.DataFrame(
        [
            {
                "file": p.name,
                "bytes": p.stat().st_size,
                "rows": file_rows[p.name],
                "target_rows": file_target_rows[p.name],
                "target_rate": file_target_rows[p.name] / file_rows[p.name] if file_rows[p.name] else 0,
            }
            for p in files
        ]
    )
    missing_df = pd.DataFrame(
        [
            {
                "column": c,
                "non_null": nonnull[c],
                "missing": total_rows - nonnull[c],
                "missing_rate": (total_rows - nonnull[c]) / total_rows,
            }
            for c in canonical
        ]
    ).sort_values("missing_rate", ascending=False)

    sampled_stats = {}
    for column, arrays in samples.items():
        values = np.concatenate(arrays) if arrays else np.array([], dtype=float)
        sampled_stats[column] = describe_series(values)
        sampled_stats[column]["global_min"] = core_min.get(column)
        sampled_stats[column]["global_max"] = core_max.get(column)
        sampled_stats[column]["quantiles_are_sampled"] = True

    summary = {
        "root": str(root),
        "file_count": len(files),
        "source_bytes": sum(p.stat().st_size for p in files),
        "rows": total_rows,
        "columns": len(canonical),
        "schema_mismatch_count": len(schema_mismatches),
        "schema_mismatches": schema_mismatches,
        "vehicle_count": len(all_vehicle_ids),
        "trip_count_sum_by_vehicle": sum(len(v) for v in vehicle_trips.values()),
        "static_rows": len(static),
        "static_unique_vehicles": int(static["VehId"].nunique()),
        "static_duplicate_ids": static_duplicate_ids,
        "matched_vehicle_count": len(matched_ids),
        "unmatched_vehicle_ids": unmatched_ids,
        "rows_with_static_match": int(vehicle_df.loc[vehicle_df["static_matched"], "rows"].sum()),
        "row_static_match_rate": float(
            vehicle_df.loc[vehicle_df["static_matched"], "rows"].sum() / total_rows
        ),
        "target_non_null": nonnull["Energy_Consumption"],
        "target_rate": nonnull["Energy_Consumption"] / total_rows,
        "energy_sign": dict(energy_sign),
        "core_sampled_statistics": sampled_stats,
        "elapsed_seconds": elapsed,
    }

    file_df.to_csv(output_dir / "file_inventory.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(output_dir / "column_missingness.csv", index=False, encoding="utf-8-sig")
    vehicle_df.to_csv(output_dir / "vehicle_summary.csv", index=False, encoding="utf-8-sig")
    grouped_df.to_csv(output_dir / "vehicle_group_summary.csv", index=False, encoding="utf-8-sig")
    static.to_csv(output_dir / "static_data_normalized.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
