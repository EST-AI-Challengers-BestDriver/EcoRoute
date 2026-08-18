"""Build offline English place labels for Ann Arbor demo road nodes.

This is a one-time data preparation step. The demo server only reads the
generated JSON file and never contacts a geocoding service at runtime.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import osmnx as ox


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecoroute.demo_server import select_spaced_nodes


OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
OSM_MAP_URL = "https://api.openstreetmap.org/api/0.6/map"
DEFAULT_OUTPUT = Path("data/processed/maps/ann_arbor/node_place_labels.json")
PLACE_TAG_KEYS = (
    "shop",
    "amenity",
    "tourism",
    "healthcare",
    "office",
    "leisure",
    "public_transport",
    "railway",
    "historic",
    "craft",
    "building",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-place-distance-m", type=float, default=300.0)
    parser.add_argument("--max-address-distance-m", type=float, default=140.0)
    parser.add_argument("--overpass-url", default=OVERPASS_URL)
    parser.add_argument("--osm-map-url", default=OSM_MAP_URL)
    return parser.parse_args()


def bbox_text(bounds: dict[str, float]) -> str:
    return ",".join(
        str(bounds[key]) for key in ("south", "west", "north", "east")
    )


def named_place_query(bounds: dict[str, float]) -> str:
    bbox = bbox_text(bounds)
    named_queries = "\n".join(
        f'  nwr["name"]["{key}"]({bbox});' for key in PLACE_TAG_KEYS
    )
    return f"""[out:json][timeout:180];
(
{named_queries}
);
out center tags;
"""


def address_query(bounds: dict[str, float]) -> str:
    return f"""[out:json][timeout:120];
nwr["addr:housenumber"]["addr:street"]({bbox_text(bounds)});
out center tags;
"""


def split_bounds(bounds: dict[str, float], parts: int = 3) -> list[dict[str, float]]:
    latitude_step = (bounds["north"] - bounds["south"]) / parts
    longitude_step = (bounds["east"] - bounds["west"]) / parts
    tiles = []
    for row in range(parts):
        for column in range(parts):
            tiles.append(
                {
                    "south": bounds["south"] + latitude_step * row,
                    "west": bounds["west"] + longitude_step * column,
                    "north": bounds["south"] + latitude_step * (row + 1),
                    "east": bounds["west"] + longitude_step * (column + 1),
                }
            )
    return tiles


def download_elements(url: str, query: str) -> list[dict[str, Any]]:
    request = Request(
        url,
        data=urlencode({"data": query}).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "User-Agent": "EcoRouteDemo/1.0 (offline Ann Arbor label builder)",
        },
        method="POST",
    )
    with urlopen(request, timeout=240) as response:
        payload = json.load(response)
    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        raise RuntimeError("Overpass response does not contain an element list")
    return elements


def has_relevant_tags(tags: dict[str, str]) -> bool:
    has_named_place = bool(tags.get("name")) and any(tags.get(key) for key in PLACE_TAG_KEYS)
    has_address = bool(tags.get("addr:housenumber") and tags.get("addr:street"))
    return has_named_place or has_address


def download_osm_map_elements(
    url: str,
    bounds: dict[str, float],
    cache_dir: Path,
    cache_key: str,
    depth: int = 0,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / f"{cache_key}.osm"
    if cache_path.exists():
        content = cache_path.read_bytes()
    else:
        query = urlencode({"bbox": ",".join(
            str(bounds[key]) for key in ("west", "south", "east", "north")
        )})
        request = Request(
            f"{url}?{query}",
            headers={
                "Accept": "application/xml",
                "User-Agent": "EcoRouteDemo/1.0 (offline Ann Arbor label builder)",
            },
        )
        try:
            with urlopen(request, timeout=180) as response:
                content = response.read()
        except HTTPError as error:
            if error.code != 400 or depth >= 3:
                raise
            elements: list[dict[str, Any]] = []
            for index, child in enumerate(split_bounds(bounds, parts=2), start=1):
                elements.extend(
                    download_osm_map_elements(
                        url,
                        child,
                        cache_dir,
                        f"{cache_key}_{index}",
                        depth=depth + 1,
                    )
                )
            return elements
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".osm.tmp")
        temporary.write_bytes(content)
        temporary.replace(cache_path)
    root = ElementTree.fromstring(content)

    coordinates: dict[str, tuple[float, float]] = {}
    elements: list[dict[str, Any]] = []
    for node in root.findall("node"):
        node_id = str(node.attrib["id"])
        latitude = float(node.attrib["lat"])
        longitude = float(node.attrib["lon"])
        coordinates[node_id] = (latitude, longitude)
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in node.findall("tag")}
        if has_relevant_tags(tags):
            elements.append(
                {
                    "type": "node",
                    "id": node_id,
                    "lat": latitude,
                    "lon": longitude,
                    "tags": tags,
                }
            )

    for way in root.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        if not has_relevant_tags(tags):
            continue
        way_coordinates = [
            coordinates[reference.attrib["ref"]]
            for reference in way.findall("nd")
            if reference.attrib.get("ref") in coordinates
        ]
        if not way_coordinates:
            continue
        latitude = sum(item[0] for item in way_coordinates) / len(way_coordinates)
        longitude = sum(item[1] for item in way_coordinates) / len(way_coordinates)
        elements.append(
            {
                "type": "way",
                "id": str(way.attrib["id"]),
                "center": {"lat": latitude, "lon": longitude},
                "tags": tags,
            }
        )
    return elements


def element_coordinate(element: dict[str, Any]) -> tuple[float, float] | None:
    latitude = element.get("lat")
    longitude = element.get("lon")
    if latitude is None or longitude is None:
        center = element.get("center") or {}
        latitude = center.get("lat")
        longitude = center.get("lon")
    try:
        return float(latitude), float(longitude)
    except (TypeError, ValueError):
        return None


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def feature_category(tags: dict[str, Any]) -> str:
    for key in PLACE_TAG_KEYS:
        if tags.get(key):
            return key
    return "place"


def parse_features(
    elements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    places: list[dict[str, Any]] = []
    addresses: list[dict[str, Any]] = []
    seen_places: set[tuple[str, int, int]] = set()
    seen_addresses: set[tuple[str, int, int]] = set()
    for element in elements:
        coordinate = element_coordinate(element)
        tags = element.get("tags") or {}
        if coordinate is None or not isinstance(tags, dict):
            continue
        latitude, longitude = coordinate
        english_name = clean_text(tags.get("name:en"))
        name = english_name or clean_text(tags.get("name"))
        if name:
            key = (name.casefold(), round(latitude * 100_000), round(longitude * 100_000))
            if key not in seen_places:
                seen_places.add(key)
                places.append(
                    {
                        "name": name,
                        "lat": latitude,
                        "lon": longitude,
                        "category": feature_category(tags),
                        "osm_type": str(element.get("type", "")),
                        "osm_id": str(element.get("id", "")),
                    }
                )
            continue
        house_number = clean_text(tags.get("addr:housenumber"))
        street = clean_text(tags.get("addr:street"))
        if not house_number or not street:
            continue
        label = f"{house_number} {street}"
        key = (label.casefold(), round(latitude * 100_000), round(longitude * 100_000))
        if key not in seen_addresses:
            seen_addresses.add(key)
            addresses.append({"label": label, "lat": latitude, "lon": longitude})
    return places, addresses


def distance_m(
    first_lat: float,
    first_lon: float,
    second_lat: float,
    second_lon: float,
) -> float:
    mean_latitude = math.radians((first_lat + second_lat) / 2)
    x = math.radians(second_lon - first_lon) * math.cos(mean_latitude)
    y = math.radians(second_lat - first_lat)
    return 6_371_008.8 * math.hypot(x, y)


def normalize_road_names(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        values = list(value)
    elif isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            values = list(parsed) if isinstance(parsed, (list, tuple)) else [value]
        except (SyntaxError, ValueError):
            values = [value]
    else:
        values = [value]
    return [clean_text(item) for item in values if clean_text(item)]


def node_road_label(graph: object, node_id: object) -> str:
    names: Counter[str] = Counter()
    for _, _, data in graph.in_edges(node_id, data=True):
        names.update(normalize_road_names(data.get("name")))
    for _, _, data in graph.out_edges(node_id, data=True):
        names.update(normalize_road_names(data.get("name")))
    roads = [name for name, _ in names.most_common(2)]
    return " & ".join(roads)


def nearest_feature(
    node: dict[str, Any], features: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    if not features:
        return None, math.inf
    nearest = min(
        features,
        key=lambda feature: distance_m(
            float(node["lat"]),
            float(node["lon"]),
            float(feature["lat"]),
            float(feature["lon"]),
        ),
    )
    return nearest, distance_m(
        float(node["lat"]),
        float(node["lon"]),
        float(nearest["lat"]),
        float(nearest["lon"]),
    )


def build_labels(
    graph: object,
    nodes: list[dict[str, Any]],
    places: list[dict[str, Any]],
    addresses: list[dict[str, Any]],
    max_place_distance_m: float,
    max_address_distance_m: float,
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for node in nodes:
        nearest_place, place_distance = nearest_feature(node, places)
        nearest_address, address_distance = nearest_feature(node, addresses)
        road = node_road_label(graph, node["graph_id"])
        if nearest_place is not None and place_distance <= max_place_distance_m:
            labels[str(node["id"])] = {
                "label": nearest_place["name"],
                "kind": nearest_place["category"],
                "distance_m": round(place_distance, 1),
                "road": road,
                "osm_type": nearest_place["osm_type"],
                "osm_id": nearest_place["osm_id"],
            }
        elif nearest_address is not None and address_distance <= max_address_distance_m:
            labels[str(node["id"])] = {
                "label": nearest_address["label"],
                "kind": "address",
                "distance_m": round(address_distance, 1),
                "road": road,
            }
        elif road:
            labels[str(node["id"])] = {
                "label": f"Near {road}",
                "kind": "road",
                "distance_m": 0.0,
                "road": road,
            }
    return labels


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config = json.loads((root / "config" / "demo_runtime.json").read_text(encoding="utf-8"))
    region = config["regions"]["ann_arbor"]
    graph = ox.io.load_graphml(root / region["graph_path"])
    bounds = {key: float(region["selectable_bounds"][key]) for key in ("south", "west", "north", "east")}
    nodes = select_spaced_nodes(
        graph,
        bounds,
        minimum_spacing_m=float(region.get("selectable_node_spacing_m", 400.0)),
    )
    graph_ids = {str(node_id): node_id for node_id in graph.nodes}
    for node in nodes:
        node["graph_id"] = graph_ids[str(node["id"])]

    tiles = split_bounds(bounds, parts=5)
    tile_cache_dir = root / "data" / "cache" / "ann_arbor_osm_tiles"
    elements: list[dict[str, Any]] = []
    for index, tile in enumerate(tiles, start=1):
        print(f"Downloading Ann Arbor map tile {index}/{len(tiles)}...", flush=True)
        elements.extend(
            download_osm_map_elements(
                args.osm_map_url,
                tile,
                tile_cache_dir,
                f"tile_{index:02d}",
            )
        )
    places, addresses = parse_features(elements)
    labels = build_labels(
        graph,
        nodes,
        places,
        addresses,
        max_place_distance_m=args.max_place_distance_m,
        max_address_distance_m=args.max_address_distance_m,
    )
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "OpenStreetMap contributors via OSM API",
        "bounds": bounds,
        "max_place_distance_m": args.max_place_distance_m,
        "max_address_distance_m": args.max_address_distance_m,
        "downloaded_element_count": len(elements),
        "named_place_count": len(places),
        "address_count": len(addresses),
        "selectable_node_count": len(nodes),
        "labeled_node_count": len(labels),
        "nodes": labels,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    kind_counts = Counter(item["kind"] for item in labels.values())
    print(f"Saved {len(labels):,}/{len(nodes):,} node labels to: {output}", flush=True)
    print(f"Downloaded {len(places):,} named places and {len(addresses):,} addresses", flush=True)
    print(f"Label kinds: {dict(kind_counts)}", flush=True)


if __name__ == "__main__":
    main()
