"""Dependency-free local HTTP server for the EcoRoute web demo."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import networkx as nx
import osmnx as ox

from .route_energy import load_hour_profiles, predict_route_energy


VEHICLES = {
    "compact": {
        "label": "소형차",
        "weight_kg": 1250.0,
        "engine_l": 1.6,
        "description": "가볍고 효율적인 도심형 차량",
    },
    "midsize": {
        "label": "중형차",
        "weight_kg": 1587.573295,
        "engine_l": 2.5,
        "description": "EcoRoute 기본 비교 차량",
    },
    "truck": {
        "label": "트럭",
        "weight_kg": 2267.96185,
        "engine_l": 4.8,
        "description": "중량과 배기량이 큰 화물 차량",
    },
}

SELECTABLE_NODE_SPACING_M = 400.0
DEMO_CONFIG_PATH = Path("config") / "demo_runtime.json"


class DemoApplication:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.web_root = root / "web"
        config_path = root / DEMO_CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        self.default_region_key = str(settings["default_region"])
        self.route_candidate_count = int(settings.get("route_candidate_count", 6))
        self.penalty_candidate_count = int(settings.get("penalty_candidate_count", 8))
        self.profile_cache_size = int(settings.get("profile_cache_size", 4))
        checkpoint_path = root / str(settings["model_checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        self.regions: dict[str, dict] = {}
        for region_key, region_settings in settings["regions"].items():
            graph_path = root / str(region_settings["graph_path"])
            traffic_path = root / str(region_settings["traffic_profile_path"])
            place_labels_setting = region_settings.get("place_labels_path")
            place_labels_path = (
                root / str(place_labels_setting) if place_labels_setting else None
            )
            required_paths = [graph_path, traffic_path]
            if place_labels_path is not None:
                required_paths.append(place_labels_path)
            for required_path in required_paths:
                if not required_path.exists():
                    raise FileNotFoundError(required_path)
            graph = ox.io.load_graphml(graph_path)
            selectable_bounds = {
                key: float(region_settings["selectable_bounds"][key])
                for key in ("south", "west", "north", "east")
            }
            selectable_node_spacing_m = float(
                region_settings.get(
                    "selectable_node_spacing_m", SELECTABLE_NODE_SPACING_M
                )
            )
            nodes = select_spaced_nodes(
                graph,
                selectable_bounds,
                minimum_spacing_m=selectable_node_spacing_m,
            )
            if place_labels_path is not None:
                place_labels = load_node_place_labels(place_labels_path)
                nodes = attach_node_place_labels(nodes, place_labels)
            self.regions[str(region_key)] = {
                "key": str(region_key),
                "label": str(region_settings.get("label", region_key)),
                "short_label": str(
                    region_settings.get(
                        "short_label", region_settings.get("label", region_key)
                    )
                ),
                "graph": graph,
                "traffic_path": traffic_path,
                "route_candidate_count": int(
                    region_settings.get(
                        "route_candidate_count", self.route_candidate_count
                    )
                ),
                "penalty_candidate_count": int(
                    region_settings.get(
                        "penalty_candidate_count", self.penalty_candidate_count
                    )
                ),
                "selectable_bounds": selectable_bounds,
                "selectable_node_spacing_m": selectable_node_spacing_m,
                "nodes": nodes,
                "profile_cache": OrderedDict(),
            }
        if self.default_region_key not in self.regions:
            raise ValueError(f"Unknown default region: {self.default_region_key}")
        self.route_lock = threading.Lock()

    def _region(self, region_key: str | None) -> dict:
        selected_key = region_key or self.default_region_key
        if selected_key not in self.regions:
            raise ValueError("지원하지 않는 지도 지역입니다.")
        return self.regions[selected_key]

    def _profiles_for(self, region: dict, hour: int) -> object:
        cache: OrderedDict = region["profile_cache"]
        if hour in cache:
            profiles = cache.pop(hour)
            cache[hour] = profiles
            return profiles
        profiles = load_hour_profiles(region["traffic_path"], hour)
        cache[hour] = profiles
        while len(cache) > self.profile_cache_size:
            cache.popitem(last=False)
        return profiles

    def config_payload(self, region_key: str | None = None) -> dict:
        region = self._region(region_key)
        selectable_bounds = region["selectable_bounds"]
        return {
            "region": region["key"],
            "region_label": region["label"],
            "regions": [
                {
                    "key": item["key"],
                    "label": item["label"],
                    "short_label": item["short_label"],
                }
                for item in self.regions.values()
            ],
            "center": {
                "lat": (selectable_bounds["south"] + selectable_bounds["north"]) / 2,
                "lon": (selectable_bounds["west"] + selectable_bounds["east"]) / 2,
            },
            "selectable_bounds": selectable_bounds,
            "selectable_node_spacing_m": region["selectable_node_spacing_m"],
            "original_graph_node_count": len(region["graph"]),
            "selectable_node_count": len(region["nodes"]),
            "nodes": region["nodes"],
            "vehicles": VEHICLES,
        }

    def calculate_routes(self, payload: dict) -> dict:
        region = self._region(str(payload.get("region", self.default_region_key)))
        start = self._coordinate(region, payload, "start")
        destination = self._coordinate(region, payload, "destination")
        if payload.get("start", {}).get("node_id") == payload.get("destination", {}).get("node_id"):
            raise ValueError("출발지와 목적지는 서로 다른 노드를 선택해 주세요.")
        hour = int(payload.get("hour", 8))
        if not 0 <= hour <= 23:
            raise ValueError("출발시간은 0시부터 23시 사이여야 합니다.")
        weekday = int(payload.get("weekday", 0))
        if not 0 <= weekday <= 6:
            raise ValueError("요일은 월요일부터 일요일 사이여야 합니다.")
        vehicle_key = str(payload.get("vehicle", "midsize"))
        if vehicle_key not in VEHICLES:
            raise ValueError("지원하지 않는 차종입니다.")
        vehicle = VEHICLES[vehicle_key]

        with self.route_lock:
            started = time.perf_counter()
            profiles = self._profiles_for(region, hour)
            summary = predict_route_energy(
                root=self.root,
                region_key=region["key"],
                hour=hour,
                weekday=weekday,
                start=start,
                destination=destination,
                vehicle_weight_kg=float(vehicle["weight_kg"]),
                engine_displacement_l=float(vehicle["engine_l"]),
                route_count=4,
                candidate_count=region["route_candidate_count"],
                penalty_candidate_count=region["penalty_candidate_count"],
                requested_device="auto",
                prepared_graph=region["graph"],
                prepared_profiles=profiles,
                save_diagnostics=False,
            )
            result_dir = self.root / "results" / "route_energy" / region["key"]
            geojson = json.loads((result_dir / "routes.geojson").read_text(encoding="utf-8"))
            metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
            elapsed_seconds = time.perf_counter() - started
            print(
                f"Demo routes ready for {region['key']} in {elapsed_seconds:.2f}s",
                flush=True,
            )

        routes = json.loads(summary.to_json(orient="records"))
        return {
            "region": region["key"],
            "region_label": region["label"],
            "hour": hour,
            "weekday": weekday,
            "vehicle": {"key": vehicle_key, **vehicle},
            "start": {"lat": start[0], "lon": start[1]},
            "destination": {"lat": destination[0], "lon": destination[1]},
            "routes": routes,
            "geojson": geojson,
            "carbon_scope": metadata["carbon_scope"],
        }

    def _coordinate(self, region: dict, payload: dict, name: str) -> tuple[float, float]:
        value = payload.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"{name} 노드를 지도에서 선택해 주세요.")
        try:
            latitude = float(value["lat"])
            longitude = float(value["lon"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} 좌표가 올바르지 않습니다.") from error
        bounds = region["selectable_bounds"]
        if not (
            bounds["south"] <= latitude <= bounds["north"]
            and bounds["west"] <= longitude <= bounds["east"]
        ):
            raise ValueError(f"{region['label']} 지도 범위 안의 도로 노드를 선택해 주세요.")
        return latitude, longitude


def select_spaced_nodes(
    graph: object,
    bounds: dict[str, float],
    minimum_spacing_m: float,
) -> list[dict[str, float | str]]:
    """Keep well-connected road nodes with a stable minimum visual spacing."""
    strongly_connected = max(nx.strongly_connected_components(graph), key=len)
    center_latitude_rad = math.radians((bounds["south"] + bounds["north"]) / 2)
    longitude_scale = math.cos(center_latitude_rad)
    candidates = []
    for node_id, data in graph.nodes(data=True):
        latitude = float(data["y"])
        longitude = float(data["x"])
        if node_id not in strongly_connected:
            continue
        if not (
            bounds["south"] <= latitude <= bounds["north"]
            and bounds["west"] <= longitude <= bounds["east"]
        ):
            continue
        degree = int(graph.in_degree(node_id) + graph.out_degree(node_id))
        candidates.append((-degree, str(node_id), node_id, latitude, longitude))
    candidates.sort()

    selected = []
    spatial_cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for _, _, node_id, latitude, longitude in candidates:
        x_m = (longitude - bounds["west"]) * 111_320 * longitude_scale
        y_m = (latitude - bounds["south"]) * 111_320
        cell_x = math.floor(x_m / minimum_spacing_m)
        cell_y = math.floor(y_m / minimum_spacing_m)
        far_enough = True
        for nearby_x in range(cell_x - 1, cell_x + 2):
            for nearby_y in range(cell_y - 1, cell_y + 2):
                for selected_x, selected_y in spatial_cells.get((nearby_x, nearby_y), []):
                    if math.hypot(x_m - selected_x, y_m - selected_y) < minimum_spacing_m:
                        far_enough = False
                        break
                if not far_enough:
                    break
            if not far_enough:
                break
        if not far_enough:
            continue
        selected.append({"id": str(node_id), "lat": latitude, "lon": longitude})
        spatial_cells.setdefault((cell_x, cell_y), []).append((x_m, y_m))
    return selected


def load_node_place_labels(path: Path) -> dict[str, dict[str, object]]:
    """Load an offline node-to-place mapping generated from OpenStreetMap."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_labels = payload.get("nodes", {})
    if not isinstance(raw_labels, dict):
        raise ValueError(f"Invalid node place label file: {path}")
    return {
        str(node_id): label
        for node_id, label in raw_labels.items()
        if isinstance(label, dict)
    }


def attach_node_place_labels(
    nodes: list[dict[str, float | str]],
    labels: dict[str, dict[str, object]],
) -> list[dict[str, float | str]]:
    """Attach validated English display labels to selectable road nodes."""
    enriched_nodes = []
    for node in nodes:
        enriched = dict(node)
        label = labels.get(str(node["id"]), {})
        place_label = str(label.get("label", "")).strip()
        if place_label:
            enriched["place_label"] = place_label
            enriched["place_kind"] = str(label.get("kind", "place"))
            try:
                enriched["place_distance_m"] = float(label.get("distance_m", 0.0))
            except (TypeError, ValueError):
                enriched["place_distance_m"] = 0.0
        enriched_nodes.append(enriched)
    return enriched_nodes


def make_handler(application: DemoApplication) -> type[BaseHTTPRequestHandler]:
    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "EcoRouteDemo/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/config":
                try:
                    region_key = parse_qs(parsed.query).get("region", [None])[0]
                    self._send_json(application.config_payload(region_key))
                except ValueError as error:
                    self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return
            static_files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/logo.png": ("logo.png", "image/png"),
            }
            entry = static_files.get(path)
            if entry is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            filename, content_type = entry
            file_path = application.web_root / filename
            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/routes":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 100_000:
                    raise ValueError("요청 데이터의 크기가 올바르지 않습니다.")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send_json(application.calculate_routes(payload))
            except (ValueError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as error:  # Keep the demo UI recoverable during calculation errors.
                print(f"Demo route calculation failed: {error}", flush=True)
                self._send_json(
                    {"error": "경로 계산에 실패했습니다. 다른 두 노드를 선택해 다시 시도해 주세요."},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[EcoRoute web] {format_string % args}", flush=True)

    return DemoHandler


def run_server(root: Path, host: str, port: int, open_browser: bool = True) -> None:
    application = DemoApplication(root)
    server = ThreadingHTTPServer((host, port), make_handler(application))
    url = f"http://{host}:{port}"
    print(f"EcoRoute demo ready: {url}", flush=True)
    print("Stop the server with Ctrl+C.", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("EcoRoute demo stopped.", flush=True)
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EcoRoute Washtenaw County web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run_server(root, args.host, args.port, open_browser=not args.no_browser)
