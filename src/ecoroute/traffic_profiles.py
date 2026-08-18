"""Create edge-by-hour traffic features from observed eVED segment speeds."""

from __future__ import annotations

import argparse
import ast
import json
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import osmnx as ox
import pandas as pd

from .preprocessing import PreprocessConfig, prepare_trip


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COORDINATE_COLUMNS = [
    "VehId",
    "Trip",
    "Timestamp(ms)",
    "Matchted Latitude[deg]",
    "Matched Longitude[deg]",
]

TRAFFIC_FEATURES = [
    "expected_speed_kmh",
    "speed_std_kmh",
    "stop_ratio",
    "low_speed_ratio",
]


def physical_edge_id(u: int, v: int, key: int) -> str:
    """Return a direction-agnostic edge identifier for MVP traffic aggregation."""
    first, second = sorted((int(u), int(v)))
    return f"{first}_{second}_{int(key)}"


def directed_edge_id(u: int, v: int, key: int) -> str:
    return f"{int(u)}_{int(v)}_{int(key)}"


def normalize_road_type(value: object) -> str:
    """Normalize OSM highway values loaded from lists or GraphML strings."""
    if isinstance(value, list):
        return str(value[0]) if value else "unclassified"
    text = str(value)
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (SyntaxError, ValueError):
            pass
    return text if text and text != "nan" else "unclassified"


def load_preprocess_config(root: Path) -> PreprocessConfig:
    manifest_path = root / "data" / "processed" / "preprocessing_manifest.json"
    if not manifest_path.exists():
        return PreprocessConfig()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return PreprocessConfig(**manifest["config"])


def representative_coordinates(segment: pd.DataFrame) -> dict[str, float] | None:
    latitude = pd.to_numeric(segment["Matchted Latitude[deg]"], errors="coerce").to_numpy(float)
    longitude = pd.to_numeric(segment["Matched Longitude[deg]"], errors="coerce").to_numpy(float)
    valid = np.isfinite(latitude) & np.isfinite(longitude)
    if not valid.any():
        return None
    valid_indices = np.flatnonzero(valid)
    local_distance = segment["step_distance_m"].to_numpy(float).cumsum()
    target_distance = float(local_distance[-1]) / 2 if len(local_distance) else 0.0
    representative_index = int(valid_indices[np.argmin(np.abs(local_distance[valid_indices] - target_distance))])
    first_index, last_index = int(valid_indices[0]), int(valid_indices[-1])
    return {
        "midpoint_latitude": float(latitude[representative_index]),
        "midpoint_longitude": float(longitude[representative_index]),
        "start_latitude": float(latitude[first_index]),
        "start_longitude": float(longitude[first_index]),
        "end_latitude": float(latitude[last_index]),
        "end_longitude": float(longitude[last_index]),
    }


def reconstruct_segment_coordinates(
    root: Path,
    segments: pd.DataFrame,
    output_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Recreate one representative GPS coordinate for every accepted 250m segment."""
    if output_path.exists() and not force:
        cached = pd.read_csv(output_path)
        if set(cached["segment_id"]) == set(segments["segment_id"]):
            print(f"Using cached segment coordinates: {output_path}", flush=True)
            return cached

    config = load_preprocess_config(root)
    raw_files = {
        path.name: path
        for path in (root / "data" / "raw" / "eVED" / "eVED").glob("*.csv")
    }
    keys = segments[["segment_id", "source_file", "vehicle_id", "trip_id", "segment_index"]].copy()
    coordinate_records: list[dict[str, float | int | str]] = []
    source_names = sorted(keys["source_file"].unique())
    for file_index, source_name in enumerate(source_names, start=1):
        path = raw_files.get(source_name)
        if path is None:
            raise FileNotFoundError(f"Raw eVED source file missing: {source_name}")
        wanted = keys.loc[keys["source_file"].eq(source_name)]
        wanted_lookup = {
            (int(row.vehicle_id), int(row.trip_id), int(row.segment_index)): row.segment_id
            for row in wanted.itertuples(index=False)
        }
        wanted_trip_frame = (
            wanted[["vehicle_id", "trip_id"]]
            .drop_duplicates()
            .rename(columns={"vehicle_id": "VehId", "trip_id": "Trip"})
        )
        print(
            f"Coordinates [{file_index:02d}/{len(source_names)}] {source_name} "
            f"({len(wanted):,} segments)",
            flush=True,
        )
        frame = pd.read_csv(path, usecols=COORDINATE_COLUMNS, low_memory=False)
        frame["VehId"] = pd.to_numeric(frame["VehId"], errors="coerce").astype("Int64")
        frame["Trip"] = pd.to_numeric(frame["Trip"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["VehId", "Trip"])
        wanted_trip_index = pd.MultiIndex.from_frame(wanted_trip_frame)
        frame_trip_index = pd.MultiIndex.from_frame(frame[["VehId", "Trip"]])
        frame = frame.loc[frame_trip_index.isin(wanted_trip_index)]
        for (vehicle_id, trip_id), group in frame.groupby(["VehId", "Trip"], sort=False):
            prepared = prepare_trip(group, config)
            for segment_index, segment in prepared.groupby("segment_index", sort=False):
                lookup_key = (int(vehicle_id), int(trip_id), int(segment_index))
                segment_id = wanted_lookup.get(lookup_key)
                if segment_id is None:
                    continue
                coordinates = representative_coordinates(segment)
                if coordinates is not None:
                    coordinate_records.append({"segment_id": segment_id, **coordinates})

    coordinates = pd.DataFrame(coordinate_records).drop_duplicates("segment_id")
    match_rate = len(coordinates) / len(segments) if len(segments) else 0.0
    if match_rate < 0.99:
        raise RuntimeError(f"Only {match_rate:.2%} of processed segments recovered GPS coordinates")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coordinates.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Recovered coordinates for {len(coordinates):,} segments", flush=True)
    return coordinates


def map_match_segments(
    graph: object,
    coordinates: pd.DataFrame,
    output_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Snap representative segment points to the nearest projected OSM edges."""
    if output_path.exists() and not force:
        cached = pd.read_csv(output_path)
        if set(cached["segment_id"]) == set(coordinates["segment_id"]):
            print(f"Using cached segment-edge matches: {output_path}", flush=True)
            return cached

    projected_graph = ox.projection.project_graph(graph)
    points = gpd.GeoSeries(
        gpd.points_from_xy(coordinates["midpoint_longitude"], coordinates["midpoint_latitude"]),
        crs="EPSG:4326",
    ).to_crs(projected_graph.graph["crs"])
    print(f"Map-matching {len(points):,} segment midpoints to OSM edges...", flush=True)
    nearest, distances = ox.distance.nearest_edges(
        projected_graph,
        X=points.x.to_numpy(),
        Y=points.y.to_numpy(),
        return_dist=True,
    )
    nearest_edges = [tuple(edge) for edge in nearest]
    edge_attributes: dict[tuple[int, int, int], tuple[str, float, float]] = {}
    for u, v, key in set(nearest_edges):
        data = projected_graph.edges[u, v, key]
        edge_attributes[(int(u), int(v), int(key))] = (
            normalize_road_type(data.get("highway", "unclassified")),
            float(data["length"]),
            float(data["speed_kph"]),
        )

    records = []
    for segment_id, edge, distance in zip(coordinates["segment_id"], nearest_edges, distances, strict=True):
        u, v, key = (int(edge[0]), int(edge[1]), int(edge[2]))
        road_type, edge_length, free_flow_speed = edge_attributes[(u, v, key)]
        records.append(
            {
                "segment_id": segment_id,
                "matched_u": u,
                "matched_v": v,
                "matched_key": key,
                "road_edge_id": physical_edge_id(u, v, key),
                "match_distance_m": float(distance),
                "road_type": road_type,
                "edge_length_m": edge_length,
                "free_flow_speed_kmh": free_flow_speed,
            }
        )
    matches = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False, encoding="utf-8-sig")
    return matches


def aggregate_observed_profiles(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate edge, road-class, and global hour profiles from matched segments."""
    observations = observations.copy()
    observed_limit = observations["speed_limit_kmh"].where(
        observations["speed_limit_kmh"].gt(0), observations["free_flow_speed_kmh"]
    )
    observations["speed_to_limit_ratio"] = (
        observations["avg_speed_kmh"] / observed_limit
    ).clip(0.05, 1.25)
    observations["display_congestion_percent"] = (
        1 - observations["speed_to_limit_ratio"].clip(upper=1.0)
    ) * 100

    edge_profiles = (
        observations.groupby(["road_edge_id", "hour"], observed=True)
        .agg(
            observations=("segment_id", "size"),
            trips=("trip_group_id", "nunique"),
            expected_speed_kmh=("avg_speed_kmh", "mean"),
            speed_std_kmh=("speed_std_kmh", "mean"),
            speed_to_limit_ratio=("speed_to_limit_ratio", "mean"),
            stop_ratio=("stop_ratio", "mean"),
            low_speed_ratio=("low_speed_ratio", "mean"),
            display_congestion_percent=("display_congestion_percent", "mean"),
        )
        .reset_index()
    )
    class_profiles = (
        observations.groupby(["road_type", "hour"], observed=True)
        .agg(
            observations=("segment_id", "size"),
            expected_speed_kmh=("avg_speed_kmh", "mean"),
            speed_std_kmh=("speed_std_kmh", "mean"),
            speed_to_limit_ratio=("speed_to_limit_ratio", "mean"),
            stop_ratio=("stop_ratio", "mean"),
            low_speed_ratio=("low_speed_ratio", "mean"),
            display_congestion_percent=("display_congestion_percent", "mean"),
        )
        .reset_index()
    )
    global_profiles = (
        observations.groupby("hour", observed=True)
        .agg(
            observations=("segment_id", "size"),
            expected_speed_kmh=("avg_speed_kmh", "mean"),
            speed_std_kmh=("speed_std_kmh", "mean"),
            speed_to_limit_ratio=("speed_to_limit_ratio", "mean"),
            stop_ratio=("stop_ratio", "mean"),
            low_speed_ratio=("low_speed_ratio", "mean"),
            display_congestion_percent=("display_congestion_percent", "mean"),
        )
        .reindex(range(24))
    )
    for column in global_profiles.columns:
        global_profiles[column] = global_profiles[column].fillna(global_profiles[column].mean())
    global_profiles.index.name = "hour"
    return edge_profiles, class_profiles, global_profiles.reset_index()


def graph_edges_frame(graph: object) -> pd.DataFrame:
    records = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        records.append(
            {
                "directed_edge_id": directed_edge_id(u, v, key),
                "u": int(u),
                "v": int(v),
                "key": int(key),
                "road_edge_id": physical_edge_id(u, v, key),
                "road_type": normalize_road_type(data.get("highway", "unclassified")),
                "length_m": float(data["length"]),
                "free_flow_speed_kmh": float(data["speed_kph"]),
            }
        )
    return pd.DataFrame(records)


def build_dense_edge_profiles(
    graph: object,
    edge_profiles: pd.DataFrame,
    class_profiles: pd.DataFrame,
    global_profiles: pd.DataFrame,
    minimum_edge_observations: int,
) -> pd.DataFrame:
    """Create exactly 24 traffic rows for every directed graph edge."""
    base_edges = graph_edges_frame(graph)
    hours = pd.DataFrame({"hour": range(24)})
    dense = base_edges.merge(hours, how="cross")

    exact_columns = [
        "road_edge_id",
        "hour",
        "observations",
        "expected_speed_kmh",
        "speed_std_kmh",
        "stop_ratio",
        "low_speed_ratio",
    ]
    dense = dense.merge(
        edge_profiles[exact_columns],
        on=["road_edge_id", "hour"],
        how="left",
        suffixes=("", "_edge"),
    )
    class_values = class_profiles[
        [
            "road_type",
            "hour",
            "observations",
            "speed_to_limit_ratio",
            "speed_std_kmh",
            "stop_ratio",
            "low_speed_ratio",
        ]
    ].rename(
        columns={
            "observations": "class_observations",
            "speed_to_limit_ratio": "class_speed_ratio",
            "speed_std_kmh": "class_speed_std_kmh",
            "stop_ratio": "class_stop_ratio",
            "low_speed_ratio": "class_low_speed_ratio",
        }
    )
    dense = dense.merge(class_values, on=["road_type", "hour"], how="left")
    global_values = global_profiles[
        ["hour", "speed_to_limit_ratio", "speed_std_kmh", "stop_ratio", "low_speed_ratio"]
    ].rename(
        columns={
            "speed_to_limit_ratio": "global_speed_ratio",
            "speed_std_kmh": "global_speed_std_kmh",
            "stop_ratio": "global_stop_ratio",
            "low_speed_ratio": "global_low_speed_ratio",
        }
    )
    dense = dense.merge(global_values, on="hour", how="left")

    use_edge = dense["observations"].fillna(0).ge(minimum_edge_observations)
    use_class = ~use_edge & dense["class_speed_ratio"].notna()
    fallback_ratio = dense["class_speed_ratio"].where(use_class, dense["global_speed_ratio"])
    fallback_speed = dense["free_flow_speed_kmh"] * fallback_ratio.clip(0.05, 1.25)
    dense["expected_speed_kmh"] = dense["expected_speed_kmh"].where(use_edge, fallback_speed)
    dense["speed_std_kmh"] = dense["speed_std_kmh"].where(
        use_edge,
        dense["class_speed_std_kmh"].where(use_class, dense["global_speed_std_kmh"]),
    )
    dense["stop_ratio"] = dense["stop_ratio"].where(
        use_edge,
        dense["class_stop_ratio"].where(use_class, dense["global_stop_ratio"]),
    )
    dense["low_speed_ratio"] = dense["low_speed_ratio"].where(
        use_edge,
        dense["class_low_speed_ratio"].where(use_class, dense["global_low_speed_ratio"]),
    )
    dense["profile_source"] = np.select(
        [use_edge, use_class], ["edge_observed", "road_class_fallback"], default="global_fallback"
    )
    dense["expected_speed_kmh"] = dense["expected_speed_kmh"].clip(
        lower=5.0, upper=dense["free_flow_speed_kmh"] * 1.10
    )
    dense["model_congestion_ratio"] = (
        dense["expected_speed_kmh"] / dense["free_flow_speed_kmh"]
    )
    dense["display_congestion_percent"] = (
        1 - dense["model_congestion_ratio"].clip(upper=1.0)
    ) * 100
    dense["expected_travel_time_s"] = (
        dense["length_m"] / dense["expected_speed_kmh"] * 3.6
    )
    dense["observations"] = dense["observations"].fillna(0).astype(int)
    result_columns = [
        "directed_edge_id",
        "u",
        "v",
        "key",
        "road_edge_id",
        "road_type",
        "hour",
        "length_m",
        "free_flow_speed_kmh",
        "expected_speed_kmh",
        "expected_travel_time_s",
        "speed_std_kmh",
        "model_congestion_ratio",
        "display_congestion_percent",
        "stop_ratio",
        "low_speed_ratio",
        "observations",
        "profile_source",
    ]
    return dense[result_columns].sort_values(["u", "v", "key", "hour"]).reset_index(drop=True)


def save_hourly_figure(global_profiles: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(
        global_profiles["hour"], global_profiles["expected_speed_kmh"], marker="o", color="#2563EB"
    )
    axes[0].set_ylabel("Observed average speed (km/h)")
    axes[0].set_title("eVED 24-hour traffic profile")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        global_profiles["hour"],
        global_profiles["display_congestion_percent"],
        marker="o",
        color="#DC2626",
        label="Congestion",
    )
    axes[1].plot(
        global_profiles["hour"],
        global_profiles["stop_ratio"] * 100,
        marker="o",
        color="#7C3AED",
        label="Stop ratio",
    )
    axes[1].set_xlabel("Hour (0-23)")
    axes[1].set_ylabel("Percent")
    axes[1].set_xticks(range(24))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_traffic_profiles(
    root: Path,
    region_key: str = "ann_arbor",
    force_coordinates: bool = False,
    force_matching: bool = False,
    maximum_match_distance_m: float = 75.0,
    minimum_edge_observations: int = 3,
) -> Path:
    segment_path = root / "data" / "processed" / "segments_250m" / "ice_segments_250m.csv"
    graph_path = (
        root / "data" / "processed" / "maps" / region_key / f"{region_key}_drive_enriched.graphml"
    )
    if not graph_path.exists():
        raise FileNotFoundError(f"Prepared map not found: {graph_path}")
    traffic_root = root / "data" / "processed" / "traffic"
    cache_root = root / "data" / "cache" / "traffic"
    region_dir = traffic_root / region_key
    cache_region_dir = cache_root / region_key
    result_dir = root / "results" / "traffic" / region_key
    coordinates_path = cache_root / "segment_coordinates.csv"
    matches_path = cache_region_dir / "segment_edge_matches.csv"
    region_dir.mkdir(parents=True, exist_ok=True)
    cache_region_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    segment_columns = [
        "segment_id",
        "source_file",
        "vehicle_id",
        "trip_id",
        "segment_index",
        "avg_speed_kmh",
        "speed_std_kmh",
        "speed_limit_kmh",
        "stop_ratio",
        "low_speed_ratio",
        "hour",
    ]
    segments = pd.read_csv(segment_path, usecols=segment_columns, low_memory=False)
    segments["trip_group_id"] = segments["vehicle_id"].astype(str) + "_" + segments["trip_id"].astype(str)
    coordinates = reconstruct_segment_coordinates(
        root, segments, coordinates_path, force_coordinates
    )
    graph = ox.io.load_graphml(graph_path)
    matches = map_match_segments(graph, coordinates, matches_path, force_matching)
    observations = segments.merge(matches, on="segment_id", how="inner", validate="one_to_one")
    observations = observations.loc[
        observations["match_distance_m"].le(maximum_match_distance_m)
    ].copy()
    if observations.empty:
        raise RuntimeError("No eVED segments matched the prepared map within the distance threshold")

    edge_profiles, class_profiles, global_profiles = aggregate_observed_profiles(observations)
    dense_profiles = build_dense_edge_profiles(
        graph,
        edge_profiles,
        class_profiles,
        global_profiles,
        minimum_edge_observations,
    )
    output_path = region_dir / "edge_hourly_profiles.csv"
    dense_profiles.to_csv(output_path, index=False, encoding="utf-8-sig")
    save_hourly_figure(global_profiles, result_dir / "hourly_traffic.png")

    sources = dense_profiles["profile_source"].value_counts().to_dict()
    matched_physical_edges = observations["road_edge_id"].nunique()
    graph_physical_edges = graph_edges_frame(graph)["road_edge_id"].nunique()
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "region": region_key,
        "time_bins": 24,
        "hours": list(range(24)),
        "weekday_split": False,
        "directional_observed_traffic": False,
        "direction_note": "MVP profiles share observed traffic between both directions of a physical edge",
        "input_segments": len(segments),
        "matched_segments_within_threshold": len(observations),
        "maximum_match_distance_m": maximum_match_distance_m,
        "matched_physical_edges": int(matched_physical_edges),
        "graph_physical_edges": int(graph_physical_edges),
        "physical_edge_coverage_pct": matched_physical_edges / graph_physical_edges * 100,
        "minimum_edge_observations": minimum_edge_observations,
        "directed_graph_edges": graph.number_of_edges(),
        "dense_profile_rows": len(dense_profiles),
        "expected_dense_profile_rows": graph.number_of_edges() * 24,
        "profile_sources": {str(key): int(value) for key, value in sources.items()},
        "model_feature_note": (
            "model_congestion_ratio preserves the trained feature semantics: expected speed / free-flow speed; "
            "display_congestion_percent is 100 * (1 - clipped ratio)"
        ),
        "output_path": str(output_path.relative_to(root)),
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved 24-hour edge profiles: {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 24-hour eVED traffic profiles")
    parser.add_argument("--region", choices=["ann_arbor", "washtenaw_county"], default="ann_arbor")
    parser.add_argument("--force-coordinates", action="store_true")
    parser.add_argument("--force-matching", action="store_true")
    parser.add_argument("--max-match-distance", type=float, default=75.0)
    parser.add_argument("--min-edge-observations", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build_traffic_profiles(
        root,
        region_key=args.region,
        force_coordinates=args.force_coordinates,
        force_matching=args.force_matching,
        maximum_match_distance_m=args.max_match_distance,
        minimum_edge_observations=args.min_edge_observations,
    )
