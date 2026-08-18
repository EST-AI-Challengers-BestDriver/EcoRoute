"""Detailed Energy_Consumption audit by powertrain for EcoRoute."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit_eved import load_static


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument("--sample-per-group", type=int, default=250_000)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = root / "data" / "processed" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "data" / "raw" / "eVED" / "eVED").glob("*.csv"))
    static = load_static(root).drop_duplicates("VehId").set_index("VehId")
    powertrain_map = static["powertrain"].to_dict()

    counts = defaultdict(Counter)
    sums = Counter()
    sumsq = Counter()
    minima = {}
    maxima = {}
    samples = defaultdict(list)
    sample_count = Counter()
    rng = np.random.default_rng(20260811)

    pair_stats = {
        "energy_fuel_rate": defaultdict(lambda: np.zeros(6, dtype=float)),
        "energy_hv_power": defaultdict(lambda: np.zeros(6, dtype=float)),
    }

    usecols = [
        "VehId",
        "Energy_Consumption",
        "Fuel Rate[L/hr]",
        "HV Battery Current[A]",
        "HV Battery Voltage[V]",
    ]
    for idx, path in enumerate(files, 1):
        print(f"[{idx:02d}/{len(files)}] {path.name}", flush=True)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunk_size, low_memory=False):
            chunk["powertrain"] = pd.to_numeric(chunk["VehId"], errors="coerce").map(powertrain_map)
            chunk["energy"] = pd.to_numeric(chunk["Energy_Consumption"], errors="coerce")
            chunk["fuel_rate"] = pd.to_numeric(chunk["Fuel Rate[L/hr]"], errors="coerce")
            chunk["hv_power"] = (
                pd.to_numeric(chunk["HV Battery Current[A]"], errors="coerce")
                * pd.to_numeric(chunk["HV Battery Voltage[V]"], errors="coerce")
                / 1000.0
            )

            for powertrain, group in chunk.groupby("powertrain", dropna=False, sort=False):
                key = "UNMATCHED" if pd.isna(powertrain) else str(powertrain)
                energy = group["energy"].to_numpy(dtype=float)
                valid = energy[np.isfinite(energy)]
                counts[key]["rows"] += len(group)
                counts[key]["target_non_null"] += len(valid)
                counts[key]["negative"] += int((valid < 0).sum())
                counts[key]["zero"] += int((valid == 0).sum())
                counts[key]["positive"] += int((valid > 0).sum())
                if len(valid):
                    sums[key] += float(valid.sum())
                    sumsq[key] += float(np.square(valid).sum())
                    minima[key] = min(minima.get(key, float(valid.min())), float(valid.min()))
                    maxima[key] = max(maxima.get(key, float(valid.max())), float(valid.max()))
                    remaining = max(0, args.sample_per_group - sample_count[key])
                    if remaining:
                        take = min(remaining, len(valid))
                        samples[key].append(valid[rng.choice(len(valid), take, replace=False)])
                        sample_count[key] += take

                for label, x_col in [("energy_fuel_rate", "fuel_rate"), ("energy_hv_power", "hv_power")]:
                    pair = group[["energy", x_col]].dropna().to_numpy(dtype=float)
                    if len(pair):
                        x, y = pair[:, 1], pair[:, 0]
                        pair_stats[label][key] += np.array(
                            [len(pair), x.sum(), y.sum(), np.square(x).sum(), np.square(y).sum(), (x * y).sum()]
                        )

    def correlation(acc: np.ndarray):
        n, sx, sy, sxx, syy, sxy = acc
        denom = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
        return None if n < 2 or denom <= 0 else float((n * sxy - sx * sy) / denom)

    rows = []
    for key in sorted(counts):
        c = counts[key]
        n = c["target_non_null"]
        values = np.concatenate(samples[key]) if samples[key] else np.array([], dtype=float)
        qs = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if len(values) else [np.nan] * 7
        variance = max(0.0, sumsq[key] / n - (sums[key] / n) ** 2) if n else np.nan
        rows.append(
            {
                "powertrain": key,
                "rows": c["rows"],
                "target_non_null": n,
                "target_rate": n / c["rows"] if c["rows"] else 0,
                "negative": c["negative"],
                "negative_rate_of_target": c["negative"] / n if n else 0,
                "zero": c["zero"],
                "positive": c["positive"],
                "mean": sums[key] / n if n else np.nan,
                "std": np.sqrt(variance),
                "min": minima.get(key),
                "p01": qs[0],
                "p05": qs[1],
                "p25": qs[2],
                "p50": qs[3],
                "p75": qs[4],
                "p95": qs[5],
                "p99": qs[6],
                "max": maxima.get(key),
                "energy_fuel_rate_pair_count": int(pair_stats["energy_fuel_rate"][key][0]),
                "energy_fuel_rate_correlation": correlation(pair_stats["energy_fuel_rate"][key]),
                "energy_hv_power_pair_count": int(pair_stats["energy_hv_power"][key][0]),
                "energy_hv_power_correlation": correlation(pair_stats["energy_hv_power"][key]),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "energy_by_powertrain.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
