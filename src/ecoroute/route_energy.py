"""Join traffic-aware routing with the trained EcoRoute DNN energy model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import osmnx as ox
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from shapely.geometry import mapping

from .dnn_training import EnergyMLP
from .routing import (
    DEFAULT_POINTS,
    ROUTE_COLORS,
    penalized_dijkstra_candidates,
    route_edge_frame,
    route_geometry,
    select_diverse_routes,
)
from .training import FEATURES


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TRAFFIC_WEIGHT = "traffic_travel_time"
TARGET_SEGMENT_M = 250.0
MINIMUM_FINAL_SEGMENT_M = 100.0
GASOLINE_KWH_PER_GALLON = 33.7
GASOLINE_CO2_KG_PER_GALLON = 8.887
CO2_KG_PER_KWH = GASOLINE_CO2_KG_PER_GALLON / GASOLINE_KWH_PER_GALLON
LITERS_PER_US_GALLON = 3.785411784
# Training-set medians used by the command-line predictor when no vehicle is supplied.
DEFAULT_VEHICLE_WEIGHT_KG = 1587.573295
DEFAULT_ENGINE_DISPLACEMENT_L = 2.5
EPA_GASOLINE_CO2_SOURCE = (
    "https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator"
)
EPA_GASOLINE_ENERGY_SOURCE = (
    "https://www.epa.gov/greenvehicles/unpublished-technology-learn-more-about-"
    "technology-assumptions-choose-path-tool"
)


def load_hour_profiles(path: Path, hour: int) -> pd.DataFrame:
    """Read one of 24 hours from a potentially large traffic CSV."""
    selected = []
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        rows = chunk.loc[chunk["hour"].eq(hour)]
        if not rows.empty:
            selected.append(rows)
    if not selected:
        raise ValueError(f"No traffic profiles found for hour {hour}: {path}")
    profiles = pd.concat(selected, ignore_index=True)
    if profiles.duplicated(["u", "v", "key"]).any():
        raise ValueError("Traffic profile contains duplicate directed edges for one hour")
    return profiles


def apply_traffic_profiles(graph: object, profiles: pd.DataFrame) -> object:
    """Attach selected-hour traffic features and routing weights to graph edges."""
    profile_lookup = {
        (int(row.u), int(row.v), int(row.key)): row
        for row in profiles.itertuples(index=False)
    }
    missing = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        profile = profile_lookup.get((int(u), int(v), int(key)))
        if profile is None:
            missing.append((u, v, key))
            continue
        data[TRAFFIC_WEIGHT] = float(profile.expected_travel_time_s)
        data["expected_speed_kmh"] = float(profile.expected_speed_kmh)
        data["traffic_speed_std_kmh"] = float(profile.speed_std_kmh)
        data["traffic_stop_ratio"] = float(profile.stop_ratio)
        data["traffic_low_speed_ratio"] = float(profile.low_speed_ratio)
        data["traffic_congestion_ratio"] = float(profile.model_congestion_ratio)
        data["display_congestion_percent"] = float(profile.display_congestion_percent)
        data["traffic_profile_source"] = str(profile.profile_source)
    if missing:
        raise ValueError(f"Traffic profile is missing {len(missing)} graph edges")
    return graph


def solve_traffic_routes(
    graph: object,
    start: tuple[float, float],
    destination: tuple[float, float],
    route_count: int = 4,
    candidate_count: int = 40,
    penalty_candidate_count: int | None = None,
) -> tuple[list[list[int]], list[object], int, int]:
    """Generate and diversity-filter traffic-time route candidates."""
    origin = ox.distance.nearest_nodes(graph, X=start[1], Y=start[0])
    destination_node = ox.distance.nearest_nodes(graph, X=destination[1], Y=destination[0])
    yen_candidates = list(
        ox.routing.k_shortest_paths(
            graph,
            origin,
            destination_node,
            k=candidate_count,
            weight=TRAFFIC_WEIGHT,
        )
    )
    penalty_candidates = penalized_dijkstra_candidates(
        graph,
        origin,
        destination_node,
        count=(
            max(12, route_count * 4)
            if penalty_candidate_count is None
            else penalty_candidate_count
        ),
        weight=TRAFFIC_WEIGHT,
    )
    candidates = list(
        {tuple(route): route for route in [*yen_candidates, *penalty_candidates]}.values()
    )
    candidates.sort(
        key=lambda route: float(route_edge_frame(graph, route, TRAFFIC_WEIGHT)[TRAFFIC_WEIGHT].sum())
    )
    routes, frames = select_diverse_routes(
        graph,
        candidates,
        route_count=route_count,
        weight=TRAFFIC_WEIGHT,
    )
    if len(routes) < route_count:
        raise RuntimeError(f"Only {len(routes)} viable traffic-aware routes found")
    return routes, frames, int(origin), int(destination_node)


def edge_piece(row: pd.Series, distance_m: float) -> dict[str, float]:
    edge_length = float(row["length"])
    fraction = distance_m / edge_length
    grade = float(row["grade"])
    return {
        "distance_m": distance_m,
        "travel_time_s": float(row[TRAFFIC_WEIGHT]) * fraction,
        "speed_std_kmh": float(row["traffic_speed_std_kmh"]),
        "speed_limit_kmh": float(row["speed_kph"]),
        "stop_ratio": float(row["traffic_stop_ratio"]),
        "low_speed_ratio": float(row["traffic_low_speed_ratio"]),
        "gradient": grade,
        "elevation_gain_m": max(grade * distance_m, 0.0),
        "observed_fraction": 1.0 if str(row["traffic_profile_source"]) == "edge_observed" else 0.0,
    }


def aggregate_segment_pieces(
    pieces: list[dict[str, float]],
    route_id: str,
    segment_index: int,
    vehicle_weight_kg: float,
    engine_displacement_l: float,
    hour: int,
    weekday: int,
) -> dict[str, float | int | str]:
    distance = sum(piece["distance_m"] for piece in pieces)
    travel_time = sum(piece["travel_time_s"] for piece in pieces)
    time_weights = np.array([piece["travel_time_s"] for piece in pieces], dtype=float)
    distance_weights = np.array([piece["distance_m"] for piece in pieces], dtype=float)
    speed_limit = float(
        np.average([piece["speed_limit_kmh"] for piece in pieces], weights=distance_weights)
    )
    average_speed = distance / travel_time * 3.6
    gradients = np.array([piece["gradient"] for piece in pieces], dtype=float)
    return {
        "route_id": route_id,
        "segment_index": segment_index,
        "vehicle_weight_kg": vehicle_weight_kg,
        "engine_displacement_l": engine_displacement_l,
        "distance_m": distance,
        "travel_time_s": travel_time,
        "avg_speed_kmh": average_speed,
        "speed_std_kmh": float(
            np.average([piece["speed_std_kmh"] for piece in pieces], weights=time_weights)
        ),
        "speed_limit_kmh": speed_limit,
        # The trained column name uses average speed / speed limit semantics.
        "congestion_ratio": average_speed / speed_limit if speed_limit > 0 else 1.0,
        "stop_ratio": float(
            np.average([piece["stop_ratio"] for piece in pieces], weights=time_weights)
        ),
        "low_speed_ratio": float(
            np.average([piece["low_speed_ratio"] for piece in pieces], weights=time_weights)
        ),
        "avg_gradient": float(np.average(gradients, weights=time_weights)),
        "max_gradient": float(gradients.max()),
        "min_gradient": float(gradients.min()),
        "elevation_gain_m": float(sum(piece["elevation_gain_m"] for piece in pieces)),
        "hour": hour,
        "weekday": weekday,
        "edge_observed_fraction": float(
            np.average([piece["observed_fraction"] for piece in pieces], weights=distance_weights)
        ),
    }


def segment_route_edges(
    edges: pd.DataFrame,
    route_id: str,
    vehicle_weight_kg: float,
    engine_displacement_l: float,
    hour: int,
    weekday: int,
    target_segment_m: float = TARGET_SEGMENT_M,
) -> list[dict[str, float | int | str]]:
    """Split an ordered OSM route into DNN-compatible approximately 250m chunks."""
    chunks: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] = []
    current_distance = 0.0
    for _, row in edges.iterrows():
        remaining = float(row["length"])
        if remaining <= 0:
            continue
        while remaining > 1e-9:
            capacity = target_segment_m - current_distance
            take = min(remaining, capacity)
            current.append(edge_piece(row, take))
            current_distance += take
            remaining -= take
            if current_distance >= target_segment_m - 1e-6:
                chunks.append(current)
                current = []
                current_distance = 0.0
    if current:
        if current_distance < MINIMUM_FINAL_SEGMENT_M and chunks:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return [
        aggregate_segment_pieces(
            pieces,
            route_id,
            segment_index,
            vehicle_weight_kg,
            engine_displacement_l,
            hour,
            weekday,
        )
        for segment_index, pieces in enumerate(chunks)
    ]


def load_model_and_predict(
    feature_frame: pd.DataFrame,
    checkpoint_path: Path,
    requested_device: str = "auto",
) -> tuple[np.ndarray, str]:
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else
        "cpu" if requested_device == "auto" else requested_device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if list(checkpoint["input_features"]) != FEATURES:
        raise ValueError("DNN checkpoint feature order does not match the integration code")
    raw = feature_frame[FEATURES].to_numpy(dtype=np.float32)
    imputer = checkpoint["imputer_statistics"].numpy()
    missing = ~np.isfinite(raw)
    if missing.any():
        raw[missing] = np.take(imputer, np.where(missing)[1])
    scaled = (raw - checkpoint["scaler_mean"].numpy()) / checkpoint["scaler_scale"].numpy()
    model = EnergyMLP(input_dim=len(FEATURES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        standardized = model(torch.from_numpy(scaled).to(device))
        predictions = (
            standardized * float(checkpoint["target_scale"]) + float(checkpoint["target_mean"])
        )
    return np.clip(predictions.cpu().numpy(), 0.0, None), str(device)


def add_carbon_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert gasoline-equivalent energy to direct tailpipe CO2 and rank routes."""
    result = summary.copy()
    result["gasoline_gallons_equivalent"] = (
        result["total_energy_kwh"] / GASOLINE_KWH_PER_GALLON
    )
    result["gasoline_liters_equivalent"] = (
        result["gasoline_gallons_equivalent"] * LITERS_PER_US_GALLON
    )
    result["total_co2_kg"] = result["total_energy_kwh"] * CO2_KG_PER_KWH
    result["co2_g_per_km"] = result["total_co2_kg"] * 1000 / result["distance_km"]
    result["rank_by_carbon"] = result["total_co2_kg"].rank(method="first").astype(int)
    result["is_fastest_route"] = result["rank_by_traffic_time"].eq(1)
    result["is_greenest_route"] = result["rank_by_carbon"].eq(1)
    # Kept for compatibility with the previous Route C result schema.
    result["rank_by_energy"] = result["rank_by_carbon"]
    result["is_recommended_eco_route"] = result["is_greenest_route"]
    return result


def save_route_figures(
    graph: object,
    routes: list[list[int]],
    summary: pd.DataFrame,
    result_dir: Path,
    hour: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")
    ax.text(
        0.05, 0.94, "EcoRoute route comparison", fontsize=24, fontweight="bold",
        color="#14213D", va="center",
    )
    ax.text(
        0.05, 0.885,
        f"Departure {hour:02d}:00  |  DNN energy prediction + gasoline tailpipe CO2",
        fontsize=11.5, color="#667085", va="center",
    )
    columns = [
        (0.08, "ROUTE"),
        (0.30, "CARBON EMISSIONS"),
        (0.52, "DISTANCE"),
        (0.70, "ESTIMATED TIME"),
        (0.87, "HIGHLIGHT"),
    ]
    for x, label in columns:
        ax.text(x, 0.81, label, fontsize=10, fontweight="bold", color="#667085", va="center")

    row_centers = np.linspace(0.69, 0.21, len(summary))
    row_height = 0.105
    for index, (row, y) in enumerate(zip(summary.itertuples(index=False), row_centers, strict=True)):
        selected = bool(row.is_fastest_route or row.is_greenest_route)
        if row.is_fastest_route and row.is_greenest_route:
            face_color, edge_color = "#E9F7EF", "#146C43"
        elif row.is_greenest_route:
            face_color, edge_color = "#E9F7EF", "#1F9D55"
        elif row.is_fastest_route:
            face_color, edge_color = "#EAF2FF", "#2563EB"
        else:
            face_color, edge_color = "#FFFFFF", "#D9E0EA"
        card = FancyBboxPatch(
            (0.045, y - row_height / 2), 0.91, row_height,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=3.0 if selected else 1.0,
            edgecolor=edge_color,
            facecolor=face_color,
        )
        ax.add_patch(card)
        weight = "bold" if selected else "normal"
        ax.plot([0.064, 0.064], [y - 0.029, y + 0.029], color=ROUTE_COLORS[index], linewidth=7)
        ax.text(0.08, y, row.route_id, fontsize=14, fontweight=weight, color="#14213D", va="center")
        ax.text(
            0.30, y, f"{row.total_co2_kg:.3f} kg CO2", fontsize=14,
            fontweight=weight, color="#14213D", va="center",
        )
        ax.text(
            0.52, y, f"{row.distance_km:.2f} km", fontsize=14,
            fontweight=weight, color="#14213D", va="center",
        )
        ax.text(
            0.70, y, f"{row.traffic_travel_time_min:.1f} min", fontsize=14,
            fontweight=weight, color="#14213D", va="center",
        )
        badges = []
        if row.is_greenest_route:
            badges.append("ECO")
        if row.is_fastest_route:
            badges.append("FASTEST")
        ax.text(
            0.87, y, " + ".join(badges) if badges else "-", fontsize=11.5,
            fontweight="bold" if badges else "normal",
            color=edge_color if badges else "#98A2B3", va="center",
        )

    ax.text(
        0.05, 0.095,
        f"Conversion: energy / {GASOLINE_KWH_PER_GALLON:.1f} kWh per gal x "
        f"{GASOLINE_CO2_KG_PER_GALLON:.3f} kg CO2 per gal",
        fontsize=10.5, color="#667085", va="center",
    )
    ax.text(
        0.05, 0.057,
        "Direct gasoline tailpipe CO2 only; fuel production and distribution are excluded.",
        fontsize=9.5, color="#98A2B3", va="center",
    )
    fig.savefig(
        result_dir / "route_energy_comparison.png", dpi=180,
        bbox_inches="tight", facecolor="#F7F9FC",
    )
    plt.close(fig)

    emphasized = summary["is_fastest_route"] | summary["is_greenest_route"]
    fig, ax = ox.plot_graph_routes(
        graph,
        routes,
        route_colors=ROUTE_COLORS[: len(routes)],
        route_linewidths=[7 if value else 3 for value in emphasized],
        node_size=0,
        edge_color="#C7CDD4",
        edge_linewidth=0.4,
        bgcolor="white",
        show=False,
        close=False,
    )
    legend = [
        Line2D(
            [0], [0], color=ROUTE_COLORS[index],
            linewidth=7 if row.is_fastest_route or row.is_greenest_route else 3,
            label=(
                f"{'[ECO] ' if row.is_greenest_route else ''}"
                f"{'[FASTEST] ' if row.is_fastest_route else ''}"
                f"{row.route_id} | {row.total_co2_kg:.3f} kg CO2 | "
                f"{row.distance_km:.2f} km | {row.traffic_travel_time_min:.1f} min"
            ),
        )
        for index, row in enumerate(summary.itertuples(index=False))
    ]
    ax.legend(handles=legend, loc="lower right")
    for text, row in zip(ax.get_legend().get_texts(), summary.itertuples(index=False), strict=True):
        if row.is_fastest_route or row.is_greenest_route:
            text.set_fontweight("bold")
    ax.set_title(f"EcoRoute candidates at {hour:02d}:00 | carbon, distance and time")
    fig.savefig(result_dir / "routes_with_energy.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def predict_route_energy(
    root: Path,
    region_key: str,
    hour: int,
    weekday: int,
    start: tuple[float, float],
    destination: tuple[float, float],
    vehicle_weight_kg: float | None = None,
    engine_displacement_l: float | None = None,
    route_count: int = 4,
    candidate_count: int = 40,
    requested_device: str = "auto",
    penalty_candidate_count: int | None = None,
    prepared_graph: object | None = None,
    prepared_profiles: pd.DataFrame | None = None,
    save_diagnostics: bool = True,
) -> pd.DataFrame:
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if not 0 <= weekday <= 6:
        raise ValueError("weekday must be between 0 and 6")
    graph_path = root / "data" / "processed" / "maps" / region_key / f"{region_key}_drive_enriched.graphml"
    traffic_dir = root / "data" / "processed" / "traffic" / region_key
    compressed_traffic_path = traffic_dir / "edge_hourly_profiles.csv.gz"
    plain_traffic_path = traffic_dir / "edge_hourly_profiles.csv"
    traffic_path = (
        compressed_traffic_path if compressed_traffic_path.exists() else plain_traffic_path
    )
    checkpoint_path = root / "models" / "dnn" / "best_model.pt"
    for required_path in [graph_path, traffic_path, checkpoint_path]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    if vehicle_weight_kg is None:
        vehicle_weight_kg = DEFAULT_VEHICLE_WEIGHT_KG
    if engine_displacement_l is None:
        engine_displacement_l = DEFAULT_ENGINE_DISPLACEMENT_L

    graph = prepared_graph if prepared_graph is not None else ox.io.load_graphml(graph_path)
    profiles = (
        prepared_profiles
        if prepared_profiles is not None
        else load_hour_profiles(traffic_path, hour)
    )
    graph = apply_traffic_profiles(graph, profiles)
    routes, frames, origin, destination_node = solve_traffic_routes(
        graph,
        start,
        destination,
        route_count,
        candidate_count,
        penalty_candidate_count=penalty_candidate_count,
    )
    segment_rows = []
    for route_number, frame in enumerate(frames, start=1):
        segment_rows.extend(
            segment_route_edges(
                frame,
                route_id=f"route_{route_number}",
                vehicle_weight_kg=float(vehicle_weight_kg),
                engine_displacement_l=float(engine_displacement_l),
                hour=hour,
                weekday=weekday,
            )
        )
    segments = pd.DataFrame(segment_rows)
    predictions, device = load_model_and_predict(segments, checkpoint_path, requested_device)
    segments["predicted_energy_kwh"] = predictions
    segments["predicted_co2_kg"] = segments["predicted_energy_kwh"] * CO2_KG_PER_KWH

    summary_rows = []
    for route_number, (route, frame) in enumerate(zip(routes, frames, strict=True), start=1):
        route_id = f"route_{route_number}"
        route_segments = segments.loc[segments["route_id"].eq(route_id)]
        distance = float(route_segments["distance_m"].sum())
        travel_time = float(route_segments["travel_time_s"].sum())
        energy = float(route_segments["predicted_energy_kwh"].sum())
        summary_rows.append(
            {
                "route_id": route_id,
                "rank_by_traffic_time": route_number,
                "distance_km": distance / 1000,
                "traffic_travel_time_min": travel_time / 60,
                "free_flow_travel_time_min": float(frame["travel_time"].astype(float).sum()) / 60,
                "average_speed_kmh": distance / travel_time * 3.6,
                "elevation_gain_m": float(route_segments["elevation_gain_m"].sum()),
                "dnn_segments": len(route_segments),
                "edge_observed_fraction": float(
                    np.average(
                        route_segments["edge_observed_fraction"],
                        weights=route_segments["distance_m"],
                    )
                ),
                "total_energy_kwh": energy,
                "energy_kwh_per_100km": energy / distance * 100_000,
                "node_count": len(route),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary = add_carbon_metrics(summary)

    result_dir = root / "results" / "route_energy" / region_key
    result_dir.mkdir(parents=True, exist_ok=True)
    if save_diagnostics:
        summary.to_csv(
            result_dir / "route_energy_summary.csv", index=False, encoding="utf-8-sig"
        )
        segments.to_csv(
            result_dir / "segment_energy_predictions.csv", index=False, encoding="utf-8-sig"
        )
    route_features = []
    for row, frame in zip(summary.itertuples(index=False), frames, strict=True):
        route_features.append(
            {
                "type": "Feature",
                "properties": {
                    "route_id": row.route_id,
                    "distance_km": row.distance_km,
                    "traffic_travel_time_min": row.traffic_travel_time_min,
                    "total_energy_kwh": row.total_energy_kwh,
                    "total_co2_kg": row.total_co2_kg,
                    "rank_by_carbon": row.rank_by_carbon,
                    "is_fastest_route": bool(row.is_fastest_route),
                    "is_greenest_route": bool(row.is_greenest_route),
                },
                "geometry": mapping(route_geometry(frame)),
            }
        )
    (result_dir / "routes.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": f"EcoRoute energy alternatives - {region_key}",
                "features": route_features,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if save_diagnostics:
        save_route_figures(graph, routes, summary, result_dir, hour)
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "region": region_key,
        "departure_hour": hour,
        "weekday_for_dnn": weekday,
        "traffic_weekday_split": False,
        "start": {"latitude": start[0], "longitude": start[1]},
        "destination": {"latitude": destination[0], "longitude": destination[1]},
        "origin_node": origin,
        "destination_node": destination_node,
        "vehicle_weight_kg": vehicle_weight_kg,
        "engine_displacement_l": engine_displacement_l,
        "target_segment_m": TARGET_SEGMENT_M,
        "model_checkpoint": str(checkpoint_path.relative_to(root)),
        "inference_device": device,
        "energy_unit": "kWh per route (sum of segment_energy_kwh predictions)",
        "carbon_conversion_applied": True,
        "carbon_scope": "direct gasoline tailpipe CO2 only",
        "gasoline_energy_kwh_per_us_gallon": GASOLINE_KWH_PER_GALLON,
        "gasoline_co2_kg_per_us_gallon": GASOLINE_CO2_KG_PER_GALLON,
        "co2_kg_per_energy_kwh": CO2_KG_PER_KWH,
        "carbon_formula": "total_co2_kg = total_energy_kwh / 33.7 * 8.887",
        "carbon_excludes": [
            "fuel extraction", "fuel refining", "fuel distribution", "vehicle manufacturing"
        ],
        "carbon_sources": [EPA_GASOLINE_CO2_SOURCE, EPA_GASOLINE_ENERGY_SOURCE],
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        summary[
            [
                "route_id", "total_energy_kwh", "total_co2_kg", "distance_km",
                "traffic_travel_time_min", "rank_by_carbon",
                "is_fastest_route", "is_greenest_route",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Route energy results saved to: {result_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict total DNN energy for four traffic-aware routes")
    parser.add_argument("--region", choices=sorted(DEFAULT_POINTS), default="ann_arbor")
    parser.add_argument("--hour", type=int, choices=range(24), default=8)
    parser.add_argument("--weekday", type=int, choices=range(7), default=2)
    parser.add_argument("--vehicle-weight-kg", type=float)
    parser.add_argument("--engine-displacement-l", type=float)
    parser.add_argument("--start-lat", type=float)
    parser.add_argument("--start-lon", type=float)
    parser.add_argument("--destination-lat", type=float)
    parser.add_argument("--destination-lon", type=float)
    parser.add_argument("--routes", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    defaults = DEFAULT_POINTS[args.region]
    start = (
        args.start_lat if args.start_lat is not None else defaults["start"][0],
        args.start_lon if args.start_lon is not None else defaults["start"][1],
    )
    destination = (
        args.destination_lat if args.destination_lat is not None else defaults["destination"][0],
        args.destination_lon if args.destination_lon is not None else defaults["destination"][1],
    )
    root = Path(__file__).resolve().parents[2]
    predict_route_energy(
        root=root,
        region_key=args.region,
        hour=args.hour,
        weekday=args.weekday,
        start=start,
        destination=destination,
        vehicle_weight_kg=args.vehicle_weight_kg,
        engine_displacement_l=args.engine_displacement_l,
        route_count=args.routes,
        candidate_count=args.candidates,
        requested_device=args.device,
    )
