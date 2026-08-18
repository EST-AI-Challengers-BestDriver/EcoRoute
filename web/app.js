const state = {
  config: null,
  setupMap: null,
  routesMap: null,
  nodesLayer: null,
  markers: { start: null, destination: null },
  points: { start: null, destination: null },
  pickMode: "start",
  result: null,
  routeLayers: new Map(),
  selectedRouteId: null,
  armedRouteId: null,
  loadingTimer: null,
  weeklyRecords: [],
  resultRecorded: false,
};

const routeColors = ["#356feb", "#ff7a1a", "#19a956", "#a458ec"];
const WEEKLY_STORAGE_KEY = "ecoroute-weekly-records-v2";
const weekDays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"];
const screens = [...document.querySelectorAll(".screen")];

function renderIntroStrokeText(target, options) {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const svgNode = (name, attributes = {}) => {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };
  const addCharacters = (textNode, kind) => {
    Array.from(options.text).forEach((character) => {
      const span = svgNode("tspan");
      span.dataset[kind] = "";
      span.textContent = character;
      textNode.appendChild(span);
    });
  };

  target.replaceChildren();
  target.className = "stroke-text";
  target.setAttribute("role", "img");
  target.setAttribute("aria-label", options.text);
  target.style.setProperty("--stroke-text-height", `${Math.round(options.fontSize * 1.3)}px`);

  const svg = svgNode("svg", {
    class: "stroke-text__svg",
    viewBox: `0 ${-options.fontSize} 600 ${options.fontSize * 1.3}`,
    preserveAspectRatio: "xMidYMid meet",
    "aria-hidden": "true",
  });
  const clipId = `intro-text-wipe-${Math.random().toString(36).slice(2, 9)}`;
  const defs = svgNode("defs");
  const clipPath = svgNode("clipPath", { id: clipId, clipPathUnits: "userSpaceOnUse" });
  const wipeRect = svgNode("rect", { x: 0, y: 0, width: 0, height: 0 });
  clipPath.appendChild(wipeRect);
  defs.appendChild(clipPath);
  svg.appendChild(defs);

  const commonTextStyle = (node) => {
    node.style.fontSize = `${options.fontSize}px`;
    node.style.fontWeight = String(options.fontWeight);
    node.style.letterSpacing = `${options.letterSpacing}px`;
  };
  const strokeText = svgNode("text", {
    class: "stroke-text__stroke",
    x: 0,
    y: 0,
    fill: "none",
    stroke: options.strokeColor,
    "stroke-width": options.strokeWidth,
    "stroke-linejoin": "round",
    "stroke-linecap": "round",
  });
  commonTextStyle(strokeText);
  addCharacters(strokeText, "strokeChar");

  const fillText = svgNode("text", {
    class: "stroke-text__fill",
    x: 0,
    y: 0,
    fill: options.fillColor,
    "clip-path": `url(#${clipId})`,
  });
  commonTextStyle(fillText);
  addCharacters(fillText, "fillChar");
  svg.append(strokeText, fillText);
  target.appendChild(svg);

  const startAnimation = async () => {
    if (document.fonts?.ready) {
      try { await document.fonts.ready; } catch { /* Use the active fallback font. */ }
    }
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const bounds = strokeText.getBBox();
    if (!bounds?.width) return;
    const padding = Math.max(options.strokeWidth, options.fontSize * .1);
    const box = {
      x: bounds.x - padding,
      y: bounds.y - padding,
      width: bounds.width + padding * 2,
      height: bounds.height + padding * 2,
    };
    svg.setAttribute("viewBox", `${box.x} ${box.y} ${box.width} ${box.height}`);
    wipeRect.setAttribute("x", box.x);
    wipeRect.setAttribute("y", box.y);
    wipeRect.setAttribute("width", box.width);
    wipeRect.setAttribute("height", box.height);

    const strokes = [...target.querySelectorAll("[data-stroke-char]")];
    const dash = Math.max(options.fontSize * 7, 200);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      strokes.forEach((stroke) => {
        stroke.style.strokeDasharray = String(dash);
        stroke.style.strokeDashoffset = "0";
      });
      return;
    }

    strokes.forEach((stroke, index) => {
      stroke.style.strokeDasharray = String(dash);
      stroke.style.strokeDashoffset = String(dash);
      stroke.animate(
        [{ strokeDashoffset: dash }, { strokeDashoffset: 0 }],
        {
          duration: options.drawDuration * 1000,
          delay: index * options.stagger * 1000,
          easing: "cubic-bezier(.25,.46,.45,.94)",
          fill: "forwards",
        },
      );
    });
    wipeRect.style.transformBox = "fill-box";
    wipeRect.style.transformOrigin = "left center";
    wipeRect.animate(
      [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }],
      {
        duration: Math.max(400, options.drawDuration * 500),
        delay: (options.drawDuration + options.fillDelay) * 1000,
        easing: "cubic-bezier(.45,0,.55,1)",
        fill: "both",
      },
    );
  };
  startAnimation();
}

function initializeIntroHero() {
  const hero = document.querySelector("#intro-hero");
  const skipButton = document.querySelector("#skip-intro");
  if (!hero) return;
  document.body.classList.add("intro-active");

  let dismissed = false;
  let dismissTimer = null;
  const dismiss = () => {
    if (dismissed) return;
    dismissed = true;
    window.clearTimeout(dismissTimer);
    document.body.classList.remove("intro-active");
    hero.classList.add("leaving");
    window.setTimeout(() => hero.remove(), 720);
  };

  const strokeTarget = document.querySelector("#intro-stroke-text");
  if (strokeTarget) {
    renderIntroStrokeText(strokeTarget, {
      text: "ECOROUTE",
      strokeColor: "#2f80ed",
      fillColor: "#f5fcff",
      strokeWidth: 1.6,
      drawDuration: 1.05,
      fillDelay: 0.12,
      stagger: 0.04,
      fontSize: 132,
      fontWeight: 850,
      letterSpacing: -5,
    });
  } else {
    hero.classList.add("intro-fallback");
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  dismissTimer = window.setTimeout(dismiss, reducedMotion ? 700 : 2350);
  skipButton?.addEventListener("click", dismiss, { once: true });
  hero.addEventListener("keydown", (event) => {
    if (event.key === "Escape" || event.key === "Enter" || event.key === " ") dismiss();
  });
}

function showScreen(id) {
  screens.forEach((screen) => screen.classList.toggle("active", screen.id === id));
  if (id === "setup-screen") setTimeout(() => state.setupMap?.invalidateSize(), 80);
  if (id === "routes-screen") setTimeout(() => state.routesMap?.invalidateSize(), 80);
}

function fillHours() {
  const select = document.querySelector("#departure-hour");
  const currentHour = new Date().getHours();
  for (let hour = 0; hour < 24; hour += 1) {
    const option = document.createElement("option");
    option.value = hour;
    option.textContent = `${String(hour).padStart(2, "0")}:00`;
    option.selected = hour === currentHour;
    select.append(option);
  }
}

function fillWeekdays() {
  const select = document.querySelector("#departure-weekday");
  weekDays.forEach((day, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = day;
    option.selected = index === 0;
    select.append(option);
  });
}

async function initialize() {
  fillHours();
  fillWeekdays();
  state.weeklyRecords = loadWeeklyRecords();
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error("지도 설정을 불러오지 못했습니다.");
    state.config = await response.json();
    renderRegionSelector();
    initializeSetupMap();
    bindEvents();
  } catch (error) {
    document.querySelector("#setup-error").textContent = error.message;
  }
}

function renderRegionSelector() {
  const selector = document.querySelector("#region-selector");
  selector.innerHTML = state.config.regions.map((region) => `
    <button class="region-button${region.key === state.config.region ? " active" : ""}"
      type="button" data-region="${region.key}"
      aria-pressed="${region.key === state.config.region}">${region.short_label}</button>`).join("");
  selector.querySelectorAll(".region-button").forEach((button) => {
    button.addEventListener("click", () => selectRegion(button.dataset.region));
  });
}

async function selectRegion(regionKey) {
  if (regionKey === state.config.region) return;
  const selector = document.querySelector("#region-selector");
  selector.querySelectorAll("button").forEach((button) => { button.disabled = true; });
  document.querySelector("#setup-error").textContent = "지도를 전환하고 있어요...";
  try {
    const response = await fetch(`/api/config?region=${encodeURIComponent(regionKey)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "지도 전환에 실패했습니다.");
    clearPoint("start");
    clearPoint("destination");
    if (state.setupMap) state.setupMap.remove();
    state.setupMap = null;
    state.nodesLayer = null;
    state.markers = { start: null, destination: null };
    state.config = payload;
    renderRegionSelector();
    initializeSetupMap();
    document.querySelector("#start-field strong").textContent = "지도에서 출발 노드를 선택하세요";
    document.querySelector("#destination-field strong").textContent = "지도에서 도착 노드를 선택하세요";
    setPickMode("start");
    updateSubmitState();
    document.querySelector("#setup-error").textContent = "";
  } catch (error) {
    document.querySelector("#setup-error").textContent = error.message;
    renderRegionSelector();
  }
}

function tileLayer() {
  return L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  });
}

function initializeSetupMap() {
  const { nodes } = state.config;
  const selectionBounds = boundsFromConfig();
  state.setupMap = L.map("setup-map", {
    zoomControl: false,
    preferCanvas: true,
    maxBounds: selectionBounds.pad(.06),
    maxBoundsViscosity: 1,
    maxZoom: 17,
  });
  tileLayer().addTo(state.setupMap);
  addCoverageFrame(state.setupMap, selectionBounds);
  const sharedRenderer = L.canvas({ padding: .35 });
  state.nodesLayer = L.layerGroup();
  nodes.forEach((node) => {
    L.circleMarker([node.lat, node.lon], {
      renderer: sharedRenderer,
      radius: 4,
      weight: 1.5,
      color: "#ffffff",
      fillColor: "#269fde",
      fillOpacity: .78,
      interactive: false,
    }).addTo(state.nodesLayer);
  });
  state.nodesLayer.addTo(state.setupMap);
  state.setupMap.fitBounds(selectionBounds, { padding: [24, 24] });
  state.setupMap.setMinZoom(state.setupMap.getZoom());
  state.setupMap.on("click", ({ latlng }) => {
    if (selectionBounds.contains(latlng)) selectNearestNode(latlng);
    else showTemporaryMapHint("선택 가능한 사각형 안을 눌러 주세요");
  });
}

function boundsFromConfig() {
  const bounds = state.config.selectable_bounds;
  return L.latLngBounds([bounds.south, bounds.west], [bounds.north, bounds.east]);
}

function addCoverageFrame(map, bounds) {
  const south = bounds.getSouth();
  const west = bounds.getWest();
  const north = bounds.getNorth();
  const east = bounds.getEast();
  const pad = 3;
  const maskStyle = {
    stroke: false,
    fillColor: "#dceef8",
    fillOpacity: .82,
    interactive: false,
  };
  [
    [[south - pad, west - pad], [south, east + pad]],
    [[north, west - pad], [north + pad, east + pad]],
    [[south, west - pad], [north, west]],
    [[south, east], [north, east + pad]],
  ].forEach((rectangle) => L.rectangle(rectangle, maskStyle).addTo(map));
  L.rectangle(bounds, {
    color: "#238bc5",
    weight: 2,
    opacity: .8,
    fill: false,
    interactive: false,
    dashArray: "7 6",
  }).addTo(map);
}

function showTemporaryMapHint(message) {
  document.querySelector("#setup-error").textContent = message;
  window.setTimeout(() => {
    if (document.querySelector("#setup-error").textContent === message) {
      document.querySelector("#setup-error").textContent = "";
    }
  }, 1500);
}

function nearestNode(latlng) {
  const lonScale = Math.cos(latlng.lat * Math.PI / 180);
  let nearest = null;
  let best = Infinity;
  state.config.nodes.forEach((node) => {
    const dy = node.lat - latlng.lat;
    const dx = (node.lon - latlng.lng) * lonScale;
    const score = dx * dx + dy * dy;
    if (score < best) { best = score; nearest = node; }
  });
  return nearest;
}

function selectNearestNode(latlng) {
  const node = nearestNode(latlng);
  if (!node) return;
  setPoint(state.pickMode, node);
  if (state.pickMode === "start" && !state.points.destination) setPickMode("destination");
}

function setPoint(kind, node) {
  state.points[kind] = { node_id: node.id, lat: node.lat, lon: node.lon };
  if (state.markers[kind]) state.setupMap.removeLayer(state.markers[kind]);
  const isStart = kind === "start";
  state.markers[kind] = L.circleMarker([node.lat, node.lon], {
    radius: 9, color: "#fff", weight: 3,
    fillColor: isStart ? "#22b6e8" : "#ff5961", fillOpacity: 1,
  }).addTo(state.setupMap).bindTooltip(isStart ? "출발" : "도착", {
    permanent: true, direction: "top", offset: [0, -8], className: "node-tooltip",
  });
  const field = document.querySelector(`#${kind === "start" ? "start" : "destination"}-field strong`);
  field.textContent = `Node ${node.id} · ${node.lat.toFixed(5)}, ${node.lon.toFixed(5)}`;
  updateSubmitState();
}

function setPickMode(kind) {
  state.pickMode = kind;
  document.querySelectorAll(".location-field").forEach((field) => {
    field.classList.toggle("active", field.dataset.pick === kind);
  });
}

function updateSubmitState() {
  document.querySelector("#find-routes").disabled = !(state.points.start && state.points.destination);
}

function bindEvents() {
  document.querySelectorAll("[data-pick]").forEach((button) => {
    button.addEventListener("click", () => setPickMode(button.dataset.pick));
  });
  document.querySelector("#swap-locations").addEventListener("click", swapLocations);
  document.querySelectorAll("input[name='vehicle']").forEach((input) => {
    input.addEventListener("change", () => {
      document.querySelectorAll(".vehicle-card").forEach((card) => card.classList.remove("selected"));
      input.closest(".vehicle-card").classList.add("selected");
    });
  });
  document.querySelector("#find-routes").addEventListener("click", calculateRoutes);
  document.querySelector("#view-weekly").addEventListener("click", showWeeklyReport);
  document.querySelector("#weekly-back").addEventListener("click", () => showScreen("impact-screen"));
  document.querySelectorAll("[data-go-home]").forEach((button) => button.addEventListener("click", resetDemo));
}

function swapLocations() {
  const start = state.points.start;
  const destination = state.points.destination;
  if (!start && !destination) return;
  if (start) setPoint("destination", start);
  else clearPoint("destination");
  if (destination) setPoint("start", destination);
  else clearPoint("start");
}

function clearPoint(kind) {
  state.points[kind] = null;
  if (state.markers[kind]) state.setupMap.removeLayer(state.markers[kind]);
  state.markers[kind] = null;
}

function startLoadingMessages() {
  const steps = [
    "24시간 교통 프로필을 불러오고 있어요",
    "다익스트라 기반 후보 경로를 비교하고 있어요",
    "도로를 250m 구간으로 나누고 있어요",
    "DNN이 에너지와 탄소 배출량을 추정하고 있어요",
  ];
  let index = 0;
  document.querySelector("#loading-step").textContent = steps[index];
  state.loadingTimer = setInterval(() => {
    index = (index + 1) % steps.length;
    document.querySelector("#loading-step").textContent = steps[index];
  }, 2100);
}

async function calculateRoutes() {
  document.querySelector("#setup-error").textContent = "";
  showScreen("loading-screen");
  startLoadingMessages();
  const vehicle = document.querySelector("input[name='vehicle']:checked").value;
  const hour = Number(document.querySelector("#departure-hour").value);
  const weekday = Number(document.querySelector("#departure-weekday").value);
  try {
    const response = await fetch("/api/routes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        region: state.config.region,
        start: state.points.start,
        destination: state.points.destination,
        hour,
        weekday,
        vehicle,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "경로 계산에 실패했습니다.");
    state.result = payload;
    state.resultRecorded = false;
    showScreen("routes-screen");
    renderRouteResults();
  } catch (error) {
    document.querySelector("#setup-error").textContent = error.message;
    showScreen("setup-screen");
  } finally {
    clearInterval(state.loadingTimer);
  }
}

function orderedRoutes(routes) {
  const eco = routes.find((route) => route.is_greenest_route);
  const fastest = routes.find((route) => route.is_fastest_route && route.route_id !== eco?.route_id);
  const selected = [eco, fastest].filter(Boolean);
  const rest = routes
    .filter((route) => !selected.some((item) => item.route_id === route.route_id))
    .sort((a, b) => a.total_co2_kg - b.total_co2_kg);
  return [...selected, ...rest];
}

function renderRouteResults() {
  if (state.routesMap) state.routesMap.remove();
  const { center } = state.config;
  const selectionBounds = boundsFromConfig();
  state.routesMap = L.map("routes-map", {
    zoomControl: false,
    maxBounds: selectionBounds.pad(.08),
    maxBoundsViscosity: 1,
    maxZoom: 17,
  }).setView([center.lat, center.lon], 13);
  tileLayer().addTo(state.routesMap);
  state.routeLayers.clear();

  const summaryById = new Map(state.result.routes.map((route) => [route.route_id, route]));
  const bounds = [];
  state.result.geojson.features.forEach((feature, index) => {
    const route = summaryById.get(feature.properties.route_id);
    const layer = L.geoJSON(feature, {
      style: { color: routeColors[index], weight: 5, opacity: .75, lineCap: "round", lineJoin: "round" },
    }).addTo(state.routesMap);
    layer.on("click", () => selectRoute(route.route_id, false));
    state.routeLayers.set(route.route_id, layer);
    bounds.push(layer.getBounds());
  });
  if (bounds.length) {
    const merged = bounds.slice(1).reduce((result, bound) => result.extend(bound), bounds[0]);
    state.routesMap.fitBounds(merged, { padding: [35, 35] });
  }

  const routes = orderedRoutes(state.result.routes);
  document.querySelector("#trip-summary").innerHTML = `
    <strong>${state.result.vehicle.label}</strong><span class="dot">•</span>
    <span>${weekDays[state.result.weekday]} · ${String(state.result.hour).padStart(2, "0")}:00 출발</span><span class="dot">•</span>
    <span>경로 ${routes.length}개 분석 완료</span>`;
  document.querySelector("#route-list").innerHTML = routes.map(routeCardHtml).join("");
  document.querySelectorAll(".route-card").forEach((card) => {
    card.addEventListener("click", () => {
      const repeat = state.armedRouteId === card.dataset.routeId;
      if (repeat) showImpact(card.dataset.routeId);
      else {
        selectRoute(card.dataset.routeId, true);
        state.armedRouteId = card.dataset.routeId;
      }
    });
  });
  state.armedRouteId = null;
  selectRoute(routes[0].route_id, true);
}

function routeCardHtml(route) {
  const originalIndex = Number(route.route_id.split("_")[1]) - 1;
  const badges = [
    route.is_greenest_route ? '<span class="badge eco">ECO</span>' : "",
    route.is_fastest_route ? '<span class="badge fast">FASTEST</span>' : "",
  ].join("");
  return `<button class="route-card" type="button" data-route-id="${route.route_id}">
    <span class="route-stripe" style="background:${routeColors[originalIndex]}"></span>
    <span class="route-card-content">
      <span class="route-card-top">
        <span class="route-name">${routeLabel(route)}</span>
        <span class="badges">${badges}</span>
      </span>
      <span class="route-metrics">
        <span class="metric"><small>예상시간</small><strong>${route.traffic_travel_time_min.toFixed(1)}분</strong></span>
        <span class="metric"><small>총 거리</small><strong>${route.distance_km.toFixed(2)}km</strong></span>
        <span class="metric"><small>예상 탄소배출</small><strong>약 ${route.total_co2_kg.toFixed(3)}kg</strong></span>
      </span>
      <span class="route-confirm">한 번 더 누르면 이 경로로 안내를 시작합니다 →</span>
    </span>
  </button>`;
}

function routeLabel(route) {
  if (route.is_greenest_route && route.is_fastest_route) return "예상 저탄소·최단시간 경로";
  if (route.is_greenest_route) return "예상 저탄소 경로";
  if (route.is_fastest_route) return "가장 빠른 경로";
  return `대안 경로 ${route.route_id.split("_")[1]}`;
}

function selectRoute(routeId, scrollCard) {
  state.selectedRouteId = routeId;
  state.routeLayers.forEach((layer, id) => {
    const selected = id === routeId;
    layer.setStyle({ weight: selected ? 10 : 5, opacity: selected ? 1 : .62 });
    if (selected) layer.bringToFront();
  });
  document.querySelectorAll(".route-card").forEach((card) => {
    const selected = card.dataset.routeId === routeId;
    card.classList.toggle("selected", selected);
    if (selected && scrollCard) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
}

function showImpact(routeId) {
  const chosen = state.result.routes.find((route) => route.route_id === routeId);
  const fastest = state.result.routes.find((route) => route.is_fastest_route);
  if (!chosen || !fastest) return;
  const relativePercent = fastest.total_energy_kwh > 0
    ? chosen.total_energy_kwh / fastest.total_energy_kwh * 100
    : 100;
  const reductionPercent = 100 - relativePercent;
  const absoluteChangePercent = Math.abs(reductionPercent);
  const tolerancePercent = .05;
  const isReduction = reductionPercent > tolerancePercent;
  const isIncrease = reductionPercent < -tolerancePercent;
  document.querySelector("#impact-comparison").textContent =
    `${absoluteChangePercent.toFixed(1)}%`;
  document.querySelector("#impact-comparison-label").textContent = isReduction
    ? "CO₂ 배출 절감"
    : isIncrease
      ? "CO₂ 배출 증가"
      : "CO₂ 배출 차이";
  document.querySelector("#comparison-chip").textContent = isReduction
    ? `가장 빠른 길 대비 ${absoluteChangePercent.toFixed(1)}% 낮음`
    : isIncrease
      ? `가장 빠른 길 대비 ${absoluteChangePercent.toFixed(1)}% 높음`
      : "가장 빠른 길과 동일";
  document.querySelector("#fast-carbon").textContent = "기준 100%";
  document.querySelector("#chosen-carbon").textContent = `${relativePercent.toFixed(1)}%`;
  const comparisonScale = Math.max(100, relativePercent, .001);
  document.querySelector("#fast-bar").style.width = `${100 / comparisonScale * 100}%`;
  document.querySelector("#chosen-bar").style.width = `${relativePercent / comparisonScale * 100}%`;
  document.querySelector("#impact-message").textContent = isReduction
    ? `가장 빠른 길보다 CO₂ 배출을 ${absoluteChangePercent.toFixed(1)}% 줄이는 경로를 선택했습니다.`
    : isIncrease
      ? `선택한 경로의 CO₂ 배출은 가장 빠른 길보다 ${absoluteChangePercent.toFixed(1)}% 높습니다.`
      : "선택한 경로와 가장 빠른 길의 CO₂ 배출은 동일합니다.";
  document.querySelector("#impact-route-meta").innerHTML = `
    <span>선택 경로 <strong>${routeLabel(chosen)}</strong></span>
    <span>요일 <strong>${weekDays[state.result.weekday]}</strong></span>
    <span>거리 <strong>${chosen.distance_km.toFixed(2)}km</strong></span>
    <span>예상시간 <strong>${chosen.traffic_travel_time_min.toFixed(1)}분</strong></span>`;
  if (!state.resultRecorded) {
    recordWeeklyResult(chosen, fastest, reductionPercent);
    state.resultRecorded = true;
  }
  document.querySelector("#view-weekly").innerHTML =
    `주간 기록 보기 (${state.weeklyRecords.length}/7) <span>→</span>`;
  showScreen("impact-screen");
}

function loadWeeklyRecords() {
  try {
    const stored = JSON.parse(sessionStorage.getItem(WEEKLY_STORAGE_KEY) || "[]");
    if (!Array.isArray(stored)) return [];
    const latestByDay = new Map();
    stored
      .filter((record) => (
        Number.isInteger(Number(record?.dayIndex))
        && Number(record.dayIndex) >= 0
        && Number(record.dayIndex) < weekDays.length
        && Number.isFinite(record?.baselineEnergy)
        && Number.isFinite(record?.chosenEnergy)
        && Number.isFinite(record?.reductionPercent)
      ))
      .forEach((record) => {
        const dayIndex = Number(record.dayIndex);
        latestByDay.set(dayIndex, { ...record, dayIndex, day: weekDays[dayIndex] });
      });
    return [...latestByDay.values()].sort((a, b) => a.dayIndex - b.dayIndex);
  } catch {
    return [];
  }
}

function saveWeeklyRecords() {
  try {
    sessionStorage.setItem(WEEKLY_STORAGE_KEY, JSON.stringify(state.weeklyRecords));
  } catch {
    // The demo still works when browser storage is unavailable; only the weekly history is lost.
  }
}

function recordWeeklyResult(chosen, fastest, reductionPercent) {
  const dayIndex = Number(state.result.weekday);
  const latestRecord = {
    dayIndex,
    day: weekDays[dayIndex],
    reductionPercent,
    baselineEnergy: fastest.total_energy_kwh,
    chosenEnergy: chosen.total_energy_kwh,
    region: state.result.region,
    regionLabel: state.result.region_label || state.result.region,
    routeLabel: routeLabel(chosen),
    distanceKm: chosen.distance_km,
  };
  state.weeklyRecords = state.weeklyRecords
    .filter((record) => Number(record.dayIndex) !== dayIndex);
  state.weeklyRecords.push(latestRecord);
  state.weeklyRecords.sort((a, b) => a.dayIndex - b.dayIndex);
  saveWeeklyRecords();
}

function showWeeklyReport() {
  renderWeeklyReport();
  showScreen("weekly-screen");
}

function renderWeeklyReport() {
  state.weeklyRecords = loadWeeklyRecords();
  const completed = state.weeklyRecords.length;
  const recordsByDay = new Map(
    state.weeklyRecords.map((record, index) => [record.dayIndex ?? index, record])
  );
  const maxMagnitude = Math.max(
    10,
    ...state.weeklyRecords.map((record) => Math.abs(record.reductionPercent))
  );
  document.querySelector("#weekly-chart").innerHTML = weekDays.map((day, index) => {
    const record = recordsByDay.get(index);
    if (!record) {
      return `<div class="weekly-day">
        <span class="weekly-day-value pending">대기</span>
        <div class="weekly-bar-track"><div class="weekly-day-bar pending"></div></div>
        <span class="weekly-day-label">${day.slice(0, 1)}<small>미기록</small></span>
      </div>`;
    }
    const value = Number(record.reductionPercent);
    const height = Math.max(4, Math.abs(value) / maxMagnitude * 100);
    const negativeClass = value < 0 ? " negative" : "";
    return `<div class="weekly-day" title="${record.regionLabel} · ${record.routeLabel}">
      <span class="weekly-day-value${negativeClass}">${value.toFixed(1)}%</span>
      <div class="weekly-bar-track"><div class="weekly-day-bar${negativeClass}" style="height:${height}%"></div></div>
      <span class="weekly-day-label">${day.slice(0, 1)}<small>완료</small></span>
    </div>`;
  }).join("");

  const baselineTotal = state.weeklyRecords.reduce(
    (total, record) => total + Number(record.baselineEnergy), 0
  );
  const chosenTotal = state.weeklyRecords.reduce(
    (total, record) => total + Number(record.chosenEnergy), 0
  );
  const weeklyReduction = baselineTotal > 0
    ? (1 - chosenTotal / baselineTotal) * 100
    : 0;
  const averageReduction = completed
    ? state.weeklyRecords.reduce(
      (total, record) => total + Number(record.reductionPercent), 0
    ) / completed
    : 0;
  const bestRecord = completed
    ? state.weeklyRecords.reduce((best, record) => (
      record.reductionPercent > best.reductionPercent ? record : best
    ))
    : null;

  document.querySelector("#weekly-progress").textContent = `${completed} / 7일`;
  document.querySelector("#weekly-total-label").textContent = weeklyReduction < 0
    ? "주간 총 CO₂ 증가율"
    : "주간 총 CO₂ 절감률";
  document.querySelector("#weekly-total").textContent = `${Math.abs(weeklyReduction).toFixed(1)}%`;
  document.querySelector("#weekly-total").closest(".weekly-summary")
    .classList.toggle("increase", weeklyReduction < 0);
  const averageElement = document.querySelector("#weekly-average");
  averageElement.textContent = `${averageReduction.toFixed(1)}%`;
  averageElement.classList.toggle("negative", averageReduction < 0);
  document.querySelector("#weekly-best").textContent = bestRecord?.day || "-";
  document.querySelector("#weekly-best-value").textContent = bestRecord
    ? `${bestRecord.reductionPercent.toFixed(1)}% · ${bestRecord.regionLabel}`
    : "아직 기록이 없습니다";
  document.querySelector("#weekly-next-route").innerHTML = completed >= 7
    ? '요일별 기록 수정하기 <span>↻</span>'
    : '다른 요일 경로 찾기 <span>→</span>';
}

function resetDemo() {
  state.result = null;
  state.resultRecorded = false;
  state.selectedRouteId = null;
  state.armedRouteId = null;
  clearPoint("start");
  clearPoint("destination");
  document.querySelector("#start-field strong").textContent = "지도에서 출발 노드를 선택하세요";
  document.querySelector("#destination-field strong").textContent = "지도에서 도착 노드를 선택하세요";
  setPickMode("start");
  updateSubmitState();
  showScreen("setup-screen");
}

initializeIntroHero();
initialize();
