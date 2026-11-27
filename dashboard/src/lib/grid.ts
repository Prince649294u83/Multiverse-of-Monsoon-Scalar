import type { Feature, FeatureCollection, Geometry, Polygon, Point } from "geojson";
import type { CellSelection, Field, ShockPreset, Variable } from "../types";
import naturalEarthCountriesRaw from "../data/natural-earth-countries.geojson?raw";

export const NLAT = 32;
export const NLON = 36;
export const LAT_MIN = 6.5;
export const LON_MIN = 65.5;
export const LAT_STEP = 1;
export const LON_STEP = 1;

export const countryContext = JSON.parse(naturalEarthCountriesRaw) as FeatureCollection;

export function cellCenter(row: number, col: number): CellSelection {
  return { row, col, lat: LAT_MIN + row * LAT_STEP, lon: LON_MIN + col * LON_STEP };
}

export function valuesForVariable(state: number[][][], variable: Variable): Field {
  const out: Field = Array.from({ length: NLAT }, () => Array(NLON).fill(0));
  for (let row = 0; row < NLAT; row += 1) {
    for (let col = 0; col < NLON; col += 1) {
      const index = variable === "t2m" ? 0 : variable === "tp" ? 1 : variable === "u10" ? 2 : variable === "v10" ? 3 : variable === "msl" ? 4 : 5;
      out[row][col] = state[index][row][col];
    }
  }
  return out;
}

function polygonForCell(row: number, col: number): Polygon {
  const lat = LAT_MIN + row * LAT_STEP;
  const lon = LON_MIN + col * LON_STEP;
  return { type: "Polygon", coordinates: [[[lon - 0.5, lat - 0.5], [lon + 0.5, lat - 0.5], [lon + 0.5, lat + 0.5], [lon - 0.5, lat + 0.5], [lon - 0.5, lat - 0.5]]] };
}

function pointInPolygon(lon: number, lat: number, polygon: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0]; const yi = polygon[i][1];
    const xj = polygon[j][0]; const yj = polygon[j][1];
    const intersect = yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

const INDIA_RING = [[68, 23], [70, 20], [72, 18], [73, 14], [76, 8], [79, 9], [82, 12], [86, 7], [90, 8], [93, 13], [96, 18], [99, 22], [97, 27], [94, 28], [91, 27], [88, 29], [85, 28], [82, 30], [79, 32], [76, 35], [73, 34], [70, 31], [68, 27], [68, 23]];

const indiaGeometry = countryContext.features.find((feature) => feature.properties?.NAME === "India")?.geometry;

function containsRing(lon: number, lat: number, ring: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i]; const [xj, yj] = ring[j];
    const intersects = yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function containsPolygon(lon: number, lat: number, polygon: number[][][]): boolean {
  return containsRing(lon, lat, polygon[0]) && !polygon.slice(1).some((hole) => containsRing(lon, lat, hole));
}

function pointIsInIndia(lon: number, lat: number): boolean {
  const geometry = indiaGeometry as Geometry | undefined;
  if (!geometry) return containsRing(lon, lat, INDIA_RING);
  if (geometry.type === "Polygon") return containsPolygon(lon, lat, geometry.coordinates as number[][][]);
  if (geometry.type === "MultiPolygon") return geometry.coordinates.some((polygon) => containsPolygon(lon, lat, polygon as number[][][]));
  return false;
}

export function isIndiaGridCell(row: number, col: number): boolean {
  return pointIsInIndia(LON_MIN + col * LON_STEP, LAT_MIN + row * LAT_STEP);
}

export const indiaOutline: Feature<Polygon> = {
  type: "Feature",
  properties: { name: "India" },
  geometry: { type: "Polygon", coordinates: [INDIA_RING] },
};

export function colorForValue(value: number, variable: Variable): string {
  if (variable === "tp") {
    if (value < 5) return "#b9f1df";
    if (value < 20) return "#43d4d0";
    if (value < 50) return "#278bd2";
    if (value < 100) return "#244aa7";
    return "#162469";
  }
  if (variable === "t2m") {
    const celsius = value - 273.15;
    if (celsius < 15) return "#2d79c8";
    if (celsius < 25) return "#45c8bd";
    if (celsius < 32) return "#f1d271";
    if (celsius < 40) return "#ee8a4e";
    return "#cf493d";
  }
  if (variable === "u10" || variable === "v10") {
    const normalized = clamp(value / 12, -1, 1);
    if (normalized < -0.66) return "#2855a4";
    if (normalized < -0.25) return "#82afd9";
    if (normalized < 0.25) return "#f1f5f1";
    if (normalized < 0.66) return "#f0a47f";
    return "#b84338";
  }
  if (variable === "msl") {
    const hpa = value / 100;
    if (hpa < 990) return "#6540ae";
    if (hpa < 1000) return "#9b7de0";
    if (hpa < 1010) return "#d5c8f4";
    if (hpa < 1020) return "#f0e4c6";
    return "#c69b52";
  }
  if (value < 15) return "#e7faff";
  if (value < 30) return "#8bd9ef";
  if (value < 50) return "#3eafd8";
  return "#1c72b5";
}

export function selectionFeature(selection: CellSelection | null): FeatureCollection<Point> {
  return {
    type: "FeatureCollection",
    features: selection ? [{ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [selection.lon, selection.lat] } }] : [],
  };
}

export function shockRegionFeature(preset: ShockPreset, centerLat: number, centerLon: number, radius: number): FeatureCollection<Polygon> {
  let ring: number[][];
  if (preset === "warm_ocean") ring = [[82, 8], [94, 8], [94, 20], [82, 20], [82, 8]];
  else if (preset === "monsoon_stall") ring = [[65, 8], [75, 8], [75, 20], [65, 20], [65, 8]];
  else if (preset === "custom") {
    ring = Array.from({ length: 49 }, (_, index) => {
      const angle = (2 * Math.PI * index) / 48;
      return [centerLon + (radius * Math.cos(angle)) / Math.max(Math.cos((centerLat * Math.PI) / 180), 0.25), centerLat + radius * Math.sin(angle)];
    });
  } else ring = [];
  return { type: "FeatureCollection", features: ring.length ? [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [ring] } }] : [] };
}

export function fieldMean(field: Field, landSeaMask?: Field, indiaOnly = false): number {
  let sum = 0;
  let totalWeight = 0;
  for (let row = 0; row < field.length; row += 1) {
    const lat = LAT_MIN + row * LAT_STEP;
    const latitudeWeight = Math.cos((lat * Math.PI) / 180);
    for (let col = 0; col < field[row].length; col += 1) {
      const landWeight = landSeaMask ? landSeaMask[row]?.[col] ?? 0 : 1;
      if (indiaOnly && (!pointIsInIndia(LON_MIN + col * LON_STEP, lat) || landWeight < 0.5)) continue;
      const weight = latitudeWeight * landWeight;
      sum += field[row][col] * weight;
      totalWeight += weight;
    }
  }
  return sum / Math.max(totalWeight, 1e-12);
}

export function deltaColor(value: number, maxAbs: number): string {
  const limit = Math.max(maxAbs, 1e-9);
  const normalized = value / limit;
  if (normalized <= -0.7) return "#2457a7";
  if (normalized <= -0.35) return "#4c86cb";
  if (normalized < -0.1) return "#a8c9e8";
  if (normalized < 0.1) return "#f1f5f1";
  if (normalized < 0.35) return "#f4c4b8";
  if (normalized < 0.7) return "#df806f";
  return "#b63839";
}

export function differenceLimit(variable: Variable): number {
  if (variable === "t2m") return 3;
  if (variable === "tp") return 15;
  if (variable === "u10" || variable === "v10") return 5;
  if (variable === "msl") return 500;
  return 10;
}

export function gridFeatureCollection(field: Field, variable: Variable, scale: "absolute" | "difference" | "accumulated" | "impact" = "absolute", differenceScale?: number, impactScale?: number, diverging = false): FeatureCollection<Polygon> {
  const features: Feature<Polygon>[] = [];
  const maxAbs = scale === "difference" ? Math.max(differenceScale ?? differenceLimit(variable), 1e-9) : 0;
  for (let row = 0; row < NLAT; row += 1) {
    for (let col = 0; col < NLON; col += 1) {
      const center = cellCenter(row, col);
      const value = field[row][col];
      const color = scale === "difference" ? deltaColor(value, maxAbs) : scale === "accumulated" ? accumulatedRainColor(value) : scale === "impact" ? impactColor(value, impactScale ?? 1, diverging) : colorForValue(value, variable);
      features.push({
        type: "Feature",
        id: row * NLON + col,
        properties: { row, col, value, color, lat: center.lat, lon: center.lon, variable, land: pointIsInIndia(center.lon, center.lat) },
        geometry: polygonForCell(row, col),
      });
    }
  }
  return { type: "FeatureCollection", features };
}

function impactColor(value: number, scale: number, diverging: boolean): string {
  const n = clamp(value / Math.max(scale, 1e-9), diverging ? -1 : 0, 1);
  if (diverging) {
    if (n <= -0.7) return "#315f9f";
    if (n <= -0.25) return "#8cb6d5";
    if (n < 0.25) return "#f1f5f1";
    if (n < 0.7) return "#e5a18f";
    return "#b84e44";
  }
  if (n < 0.2) return "#e8f6ee";
  if (n < 0.45) return "#b0e1ce";
  if (n < 0.7) return "#58b9a0";
  if (n < 0.9) return "#3383b7";
  return "#3f428d";
}

function accumulatedRainColor(value: number): string {
  if (value < 25) return "#e8f6ee";
  if (value < 75) return "#b0e1ce";
  if (value < 150) return "#58b9a0";
  if (value < 250) return "#3383b7";
  if (value < 400) return "#4f54a2";
  return "#3b286e";
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function formatValue(value: number, variable: Variable): string {
  if (variable === "t2m") return `${(value - 273.15).toFixed(1)} °C`;
  if (variable === "msl") return `${(value / 100).toFixed(1)} hPa`;
  if (variable === "tp") return `${value.toFixed(1)} mm/day`;
  if (variable === "tcwv") return `${value.toFixed(1)} kg/m²`;
  return `${value.toFixed(1)} m/s`;
}

export function unitFor(variable: Variable): string {
  if (variable === "t2m") return "°C";
  if (variable === "msl") return "hPa";
  if (variable === "tp") return "mm/day";
  if (variable === "tcwv") return "kg/m²";
  if (variable === "u10") return "m/s · eastward";
  if (variable === "v10") return "m/s · northward";
  return "kg/m²";
}
