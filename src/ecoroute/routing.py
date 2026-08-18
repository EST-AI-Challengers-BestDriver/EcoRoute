"""Dijkstra/Yen-based diverse route generation for EcoRoute."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import osmnx as ox
from shapely.geometry import MultiLineString, mapping


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_POINTS = {
    "ann_arbor": {
        "start": (42.26587, -83.74873),  # Michigan Stadium
        "destination": (42.31553, -83.68061),  # Domino's Farms area
    },
    "washtenaw_county": {
        "start": (42.31810, -84.02050),  # Chelsea
        "destination": (42.24110, -83.61300),  # Ypsilanti
    },
}

ROUTE_COLORS = ["#2563EB", "#F97316", "#16A34A", "#A855F7"]


def route_edge_frame(graph: object, route: list[int], weight: str = "travel_time") -> object:
    return ox.routing.route_to_gdf(graph, route, weight=weight)


def penalized_dijkstra_candidates(
    graph: object,
    origin: int,
    destination: int,
    count: int = 16,
    penalty_factor: float = 1.5,
    weight: str = "travel_time",
) -> list[list[int]]:
    """Generate alternatives by penalizing roads reused by earlier Dijkstra paths."""
    usage: dict[tuple[int, int], int] = {}
    candidates: list[list[int]] = []
    for _ in range(count):
        for u, v, _, data in graph.edges(keys=True, data=True):
            reuse_count = usage.get((int(u), int(v)), 0)
            data["diverse_weight"] = float(data[weight]) * (
                1.0 + penalty_factor * reuse_count
            )
        route = ox.routing.shortest_path(
            graph,
            origin,
            destination,
            weight="diverse_weight",
        )
        if route is None:
            break
        candidates.append(route)
        for u, v in zip(route[:-1], route[1:], strict=True):
            usage[(int(u), int(v))] = usage.get((int(u), int(v)), 0) + 1
    return candidates


def edge_overlap_ratio(first_edges: object, second_edges: object) -> float:
    """Return shared directed-edge length divided by the shorter route length."""
    first = {tuple(index): float(row["length"]) for index, row in first_edges.iterrows()}
    second = {tuple(index): float(row["length"]) for index, row in second_edges.iterrows()}
    shared = sum(min(first[index], second[index]) for index in first.keys() & second.keys())
    denominator = min(sum(first.values()), sum(second.values()))
    return shared / denominator if denominator else 0.0


def select_diverse_routes(
    graph: object,
    candidates: list[list[int]],
    route_count: int = 4,
    maximum_overlap: float = 0.85,
    maximum_cost_ratio: float = 1.8,
    weight: str = "travel_time",
) -> tuple[list[list[int]], list[object]]:
    """Greedily retain short candidates that do not mostly reuse selected roads."""
    if not candidates:
        return [], []
    frames = [route_edge_frame(graph, route, weight) for route in candidates]
    costs = [float(frame[weight].sum()) for frame in frames]
    selected_indices = [0]
    for index in range(1, len(candidates)):
        if costs[index] > costs[0] * maximum_cost_ratio:
            continue
        overlaps = [edge_overlap_ratio(frames[index], frames[chosen]) for chosen in selected_indices]
        if max(overlaps, default=0.0) <= maximum_overlap:
            selected_indices.append(index)
        if len(selected_indices) >= route_count:
            break

    if len(selected_indices) < route_count:
        remaining = []
        for index in range(1, len(candidates)):
            if index in selected_indices or costs[index] > costs[0] * maximum_cost_ratio:
                continue
            max_overlap = max(
                edge_overlap_ratio(frames[index], frames[chosen]) for chosen in selected_indices
            )
            remaining.append((max_overlap, costs[index], index))
        for _, _, index in sorted(remaining):
            selected_indices.append(index)
            if len(selected_indices) >= route_count:
                break
    return [candidates[index] for index in selected_indices], [frames[index] for index in selected_indices]


def summarize_route(graph: object, route: list[int], edges: object, route_number: int) -> dict:
    lengths = edges["length"].astype(float).to_numpy()
    grades = edges["grade"].astype(float).to_numpy()
    elevation_gain = sum(
        max(0.0, float(graph.nodes[v]["elevation"]) - float(graph.nodes[u]["elevation"]))
        for u, v in zip(route[:-1], route[1:], strict=True)
    )
    return {
        "route_id": f"route_{route_number}",
        "rank_by_free_flow_time": route_number,
        "node_count": len(route),
        "edge_count": len(edges),
        "distance_m": float(lengths.sum()),
        "free_flow_travel_time_s": float(edges["travel_time"].astype(float).sum()),
        "average_speed_kph": float(lengths.sum() / edges["travel_time"].astype(float).sum() * 3.6),
        "elevation_gain_m": float(elevation_gain),
        "average_gradient": float(np.average(grades, weights=lengths)),
        "maximum_absolute_gradient": float(np.max(np.abs(grades))),
        "nodes": [int(node) for node in route],
    }


def route_geometry(edges: object) -> MultiLineString:
    lines = []
    for geometry in edges.geometry:
        if geometry.geom_type == "LineString":
            lines.append(geometry)
        elif geometry.geom_type == "MultiLineString":
            lines.extend(geometry.geoms)
    return MultiLineString(lines)


def generate_routes(
    root: Path,
    region_key: str,
    start: tuple[float, float],
    destination: tuple[float, float],
    route_count: int = 4,
    candidate_count: int = 40,
) -> list[dict]:
    graph_path = (
        root
        / "data"
        / "processed"
        / "maps"
        / region_key
        / f"{region_key}_drive_enriched.graphml"
    )
    if not graph_path.exists():
        raise FileNotFoundError(f"Prepared map not found. Run prepare_map.py first: {graph_path}")
    graph = ox.io.load_graphml(graph_path)
    origin = ox.distance.nearest_nodes(graph, X=start[1], Y=start[0])
    destination_node = ox.distance.nearest_nodes(
        graph, X=destination[1], Y=destination[0]
    )
    print(f"Snapped nodes: origin={origin}, destination={destination_node}", flush=True)

    yen_candidates = list(
        ox.routing.k_shortest_paths(
            graph,
            origin,
            destination_node,
            k=candidate_count,
            weight="travel_time",
        )
    )
    penalty_candidates = penalized_dijkstra_candidates(
        graph,
        origin,
        destination_node,
        count=max(12, route_count * 4),
    )
    unique_candidates = {
        tuple(route): route for route in [*yen_candidates, *penalty_candidates]
    }
    candidates = list(unique_candidates.values())
    candidates.sort(
        key=lambda route: float(route_edge_frame(graph, route, "travel_time")["travel_time"].sum())
    )
    routes, frames = select_diverse_routes(
        graph,
        candidates,
        route_count=route_count,
        weight="travel_time",
    )
    if len(routes) < route_count:
        raise RuntimeError(f"Only {len(routes)} viable routes found; requested {route_count}")

    result_dir = root / "results" / "routing" / region_key
    result_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        summarize_route(graph, route, edges, index)
        for index, (route, edges) in enumerate(zip(routes, frames, strict=True), start=1)
    ]
    overlap_matrix = [
        [round(edge_overlap_ratio(first, second), 6) for second in frames]
        for first in frames
    ]
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "region": region_key,
        "algorithm": (
            "Yen K-shortest loopless paths plus repeated edge-penalized Dijkstra searches, "
            "followed by route-overlap filtering"
        ),
        "weight": "free-flow travel_time",
        "start_input": {"latitude": start[0], "longitude": start[1]},
        "destination_input": {"latitude": destination[0], "longitude": destination[1]},
        "origin_node": int(origin),
        "destination_node": int(destination_node),
        "candidate_count": len(candidates),
        "selected_route_count": len(routes),
        "directed_edge_length_overlap_matrix": overlap_matrix,
        "routes": summaries,
    }
    (result_dir / "routes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    features = []
    for summary, edges in zip(summaries, frames, strict=True):
        properties = {key: value for key, value in summary.items() if key != "nodes"}
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(route_geometry(edges)),
            }
        )
    geojson = {
        "type": "FeatureCollection",
        "name": f"EcoRoute alternatives - {region_key}",
        "features": features,
    }
    (result_dir / "routes.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False), encoding="utf-8"
    )

    fig, ax = ox.plot_graph_routes(
        graph,
        routes,
        route_colors=ROUTE_COLORS[: len(routes)],
        route_linewidths=[4] * len(routes),
        node_size=0,
        edge_color="#C7CDD4",
        edge_linewidth=0.45,
        bgcolor="white",
        show=False,
        close=False,
    )
    ax.scatter(
        [graph.nodes[origin]["x"], graph.nodes[destination_node]["x"]],
        [graph.nodes[origin]["y"], graph.nodes[destination_node]["y"]],
        c=["#111827", "#DC2626"],
        s=60,
        zorder=5,
    )
    ax.set_title(f"EcoRoute: four diverse {region_key} route candidates")
    fig.savefig(result_dir / "routes.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for summary in summaries:
        print(
            f"{summary['route_id']}: {summary['distance_m'] / 1000:.2f} km, "
            f"{summary['free_flow_travel_time_s'] / 60:.1f} min, "
            f"elevation gain {summary['elevation_gain_m']:.1f} m",
            flush=True,
        )
    print(f"Routing results saved to: {result_dir}", flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate four EcoRoute alternatives")
    parser.add_argument("--region", choices=sorted(DEFAULT_POINTS), default="ann_arbor")
    parser.add_argument("--start-lat", type=float)
    parser.add_argument("--start-lon", type=float)
    parser.add_argument("--destination-lat", type=float)
    parser.add_argument("--destination-lon", type=float)
    parser.add_argument("--routes", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=40)
    args = parser.parse_args()
    defaults = DEFAULT_POINTS[args.region]
    start = (
        args.start_lat if args.start_lat is not None else defaults["start"][0],
        args.start_lon if args.start_lon is not None else defaults["start"][1],
    )
    destination = (
        args.destination_lat
        if args.destination_lat is not None
        else defaults["destination"][0],
        args.destination_lon
        if args.destination_lon is not None
        else defaults["destination"][1],
    )
    root = Path(__file__).resolve().parents[2]
    generate_routes(
        root,
        args.region,
        start,
        destination,
        route_count=args.routes,
        candidate_count=args.candidates,
    )
