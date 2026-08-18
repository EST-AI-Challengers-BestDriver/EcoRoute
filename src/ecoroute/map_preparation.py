"""Prepare OSM driving graphs and SRTM/Mapzen HGT elevation for EcoRoute."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import requests


REGIONS = {
    "ann_arbor": {
        "place": "Ann Arbor, Michigan, USA",
        "buffer_m": 3_000,
    },
    "washtenaw_county": {
        "place": "Washtenaw County, Michigan, USA",
        "buffer_m": 1_000,
    },
}

HIGHWAY_SPEEDS_KPH = {
    "motorway": 105,
    "motorway_link": 72,
    "trunk": 90,
    "trunk_link": 64,
    "primary": 72,
    "primary_link": 56,
    "secondary": 56,
    "secondary_link": 48,
    "tertiary": 48,
    "tertiary_link": 40,
    "residential": 40,
    "living_street": 16,
    "service": 24,
    "unclassified": 40,
}

HGT_BASE_URL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"


def hgt_tile_name(latitude: float, longitude: float) -> str:
    """Return the one-degree HGT tile containing a WGS84 coordinate."""
    lat_floor = math.floor(latitude)
    lon_floor = math.floor(longitude)
    lat_part = f"{'N' if lat_floor >= 0 else 'S'}{abs(lat_floor):02d}"
    lon_part = f"{'E' if lon_floor >= 0 else 'W'}{abs(lon_floor):03d}"
    return f"{lat_part}{lon_part}"


def required_hgt_tiles(graph: object) -> list[str]:
    latitudes = [float(data["y"]) for _, data in graph.nodes(data=True)]
    longitudes = [float(data["x"]) for _, data in graph.nodes(data=True)]
    tiles = {
        hgt_tile_name(latitude, longitude)
        for latitude in [min(latitudes), max(latitudes)]
        for longitude in [min(longitudes), max(longitudes)]
    }
    for lat_degree in range(math.floor(min(latitudes)), math.floor(max(latitudes)) + 1):
        for lon_degree in range(math.floor(min(longitudes)), math.floor(max(longitudes)) + 1):
            tiles.add(hgt_tile_name(lat_degree + 0.5, lon_degree + 0.5))
    return sorted(tiles)


def download_hgt_tile(tile_name: str, dem_dir: Path) -> Path:
    """Download one gzip-compressed HGT tile, reusing the local cache."""
    dem_dir.mkdir(parents=True, exist_ok=True)
    destination = dem_dir / f"{tile_name}.hgt.gz"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    latitude_folder = tile_name[:3]
    url = f"{HGT_BASE_URL}/{latitude_folder}/{tile_name}.hgt.gz"
    print(f"Downloading elevation tile {tile_name}...", flush=True)
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def load_hgt_tile(path: Path) -> np.ndarray:
    """Read a big-endian signed 16-bit HGT raster from gzip."""
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    sample_count = len(payload) // 2
    side = int(math.isqrt(sample_count))
    if side * side != sample_count:
        raise ValueError(f"Invalid HGT raster dimensions: {path}")
    return np.frombuffer(payload, dtype=">i2").reshape(side, side)


def interpolate_hgt(raster: np.ndarray, tile_name: str, latitude: float, longitude: float) -> float:
    """Bilinearly interpolate elevation in meters from an HGT raster."""
    lat_floor = int(tile_name[1:3]) * (1 if tile_name[0] == "N" else -1)
    lon_floor = int(tile_name[4:7]) * (1 if tile_name[3] == "E" else -1)
    side = raster.shape[0]
    row = (lat_floor + 1 - latitude) * (side - 1)
    column = (longitude - lon_floor) * (side - 1)
    row = float(np.clip(row, 0, side - 1))
    column = float(np.clip(column, 0, side - 1))
    row0, col0 = int(math.floor(row)), int(math.floor(column))
    row1, col1 = min(row0 + 1, side - 1), min(col0 + 1, side - 1)
    row_weight, col_weight = row - row0, column - col0
    values = np.array(
        [raster[row0, col0], raster[row0, col1], raster[row1, col0], raster[row1, col1]],
        dtype=float,
    )
    if np.any(values <= -32_768):
        valid = values[values > -32_768]
        return float(valid.mean()) if len(valid) else 0.0
    top = values[0] * (1 - col_weight) + values[1] * col_weight
    bottom = values[2] * (1 - col_weight) + values[3] * col_weight
    return float(top * (1 - row_weight) + bottom * row_weight)


def add_graph_elevations(graph: object, dem_dir: Path) -> tuple[object, list[str]]:
    """Attach HGT elevations to nodes and calculate directed edge grades."""
    tile_names = required_hgt_tiles(graph)
    rasters = {
        tile: load_hgt_tile(download_hgt_tile(tile, dem_dir))
        for tile in tile_names
    }
    elevations: dict[int, float] = {}
    for node, data in graph.nodes(data=True):
        latitude, longitude = float(data["y"]), float(data["x"])
        tile = hgt_tile_name(latitude, longitude)
        elevations[node] = interpolate_hgt(rasters[tile], tile, latitude, longitude)
    for node, elevation in elevations.items():
        graph.nodes[node]["elevation"] = round(elevation, 3)
    graph = ox.elevation.add_edge_grades(graph, add_absolute=True)
    return graph, tile_names


def buffered_place_polygon(place: str, buffer_m: float) -> object:
    boundary = ox.geocoder.geocode_to_gdf(place)
    projected = ox.projection.project_gdf(boundary)
    buffered = projected.geometry.union_all().buffer(buffer_m)
    return gpd.GeoSeries([buffered], crs=projected.crs).to_crs("EPSG:4326").iloc[0]


def prepare_region(root: Path, region_key: str = "ann_arbor", force: bool = False) -> Path:
    """Download, enrich, validate, and persist one regional drive graph."""
    if region_key not in REGIONS:
        raise ValueError(f"Unknown region {region_key!r}. Choose from {sorted(REGIONS)}")
    region = REGIONS[region_key]
    raw_dir = root / "data" / "raw" / "maps" / region_key
    processed_dir = root / "data" / "processed" / "maps" / region_key
    cache_dir = root / "data" / "raw" / "maps" / "osmnx_cache"
    dem_dir = raw_dir / "dem"
    raw_graph_path = raw_dir / f"{region_key}_drive_raw.graphml"
    graph_path = processed_dir / f"{region_key}_drive_enriched.graphml"
    metadata_path = processed_dir / "metadata.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir
    ox.settings.requests_timeout = 300

    if graph_path.exists() and not force:
        print(f"Prepared graph already exists: {graph_path}", flush=True)
        return graph_path

    if raw_graph_path.exists() and not force:
        print(f"Loading cached raw graph: {raw_graph_path}", flush=True)
        graph = ox.io.load_graphml(raw_graph_path)
    else:
        print(f"Downloading OSM drive network for {region['place']}...", flush=True)
        polygon = buffered_place_polygon(str(region["place"]), float(region["buffer_m"]))
        graph = ox.graph.graph_from_polygon(
            polygon,
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
        )
        ox.io.save_graphml(graph, raw_graph_path)

    graph = ox.routing.add_edge_speeds(
        graph,
        hwy_speeds=HIGHWAY_SPEEDS_KPH,
        fallback=40,
    )
    graph = ox.routing.add_edge_travel_times(graph)
    graph, elevation_tiles = add_graph_elevations(graph, dem_dir)

    if not graph.is_directed():
        raise ValueError("Prepared driving graph must be directed")
    edge_count = graph.number_of_edges()
    missing_elevation = sum("elevation" not in data for _, data in graph.nodes(data=True))
    missing_routing = sum(
        any(attribute not in data for attribute in ["length", "speed_kph", "travel_time", "grade"])
        for _, _, _, data in graph.edges(keys=True, data=True)
    )
    if missing_elevation or missing_routing:
        raise ValueError(
            f"Prepared graph has missing values: nodes={missing_elevation}, edges={missing_routing}"
        )

    ox.io.save_graphml(graph, graph_path)
    node_lats = [float(data["y"]) for _, data in graph.nodes(data=True)]
    node_lons = [float(data["x"]) for _, data in graph.nodes(data=True)]
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "region_key": region_key,
        "place": region["place"],
        "boundary_buffer_m": region["buffer_m"],
        "network_type": "drive",
        "nodes": graph.number_of_nodes(),
        "edges": edge_count,
        "bounds_wgs84": {
            "south": min(node_lats),
            "west": min(node_lons),
            "north": max(node_lats),
            "east": max(node_lons),
        },
        "elevation_tiles": elevation_tiles,
        "elevation_source": HGT_BASE_URL,
        "road_source": "OpenStreetMap via OSMnx",
        "road_data_attribution": "© OpenStreetMap contributors, ODbL 1.0",
        "graph_path": str(graph_path.relative_to(root)),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Prepared {region_key}: {graph.number_of_nodes():,} nodes, {edge_count:,} edges",
        flush=True,
    )
    print(f"Saved graph: {graph_path}", flush=True)
    return graph_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an EcoRoute OSM driving graph")
    parser.add_argument("--region", choices=sorted(REGIONS), default="ann_arbor")
    parser.add_argument("--force", action="store_true", help="Redownload and rebuild existing data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    prepare_region(root, args.region, args.force)
