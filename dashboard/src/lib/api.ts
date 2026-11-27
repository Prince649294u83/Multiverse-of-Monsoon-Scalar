import type { SeedState, SimulationRequest, SimulationResponse } from "../types";

const baseUrl = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

function validGridSequence(states: number[][][][], expectedSteps: number): boolean {
  return Array.isArray(states) && states.length === expectedSteps && states.every((step) => step.length === 6 && step.every((channel) => channel.length === 32 && channel.every((row) => row.length === 36 && row.every(Number.isFinite))));
}

async function postJson<T>(path: string, body?: unknown, signal?: AbortSignal, timeoutMs = 30_000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(new Error(`Request timed out after ${Math.round(timeoutMs / 1000)} seconds`)), timeoutMs);
  const abort = () => controller.abort(signal?.reason);
  signal?.addEventListener("abort", abort, { once: true });
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    const result = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = typeof result?.detail === "string" ? result.detail : `Inference server returned ${response.status}`;
      throw new Error(detail);
    }
    return result as T;
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}

export async function checkHealth(): Promise<{ status: string; model_name: string }> {
  const result = await postJson<{ status: string; model_name: string }>("/health", undefined, undefined, 4_000);
  if (result.status !== "ready") throw new Error("Inference server is not ready");
  return result;
}

export async function loadInitialState(doy: number, signal?: AbortSignal): Promise<SeedState> {
  const result = await postJson<{ init_states: number[][][][]; climatology_reference: number[][][][]; doy_sincos: [number, number]; land_sea_mask: number[][]; source: string }>("/initial-state", { doy }, signal, 15_000);
  if (!validGridSequence(result.init_states, 3) || result.land_sea_mask.length !== 32 || result.land_sea_mask.some((row) => row.length !== 36 || row.some((value) => !Number.isFinite(value)))) {
    throw new Error("Server returned an invalid climatology grid");
  }
  if (!validGridSequence(result.climatology_reference, 14)) {
    throw new Error("Server returned an invalid climatology reference grid");
  }
  return { initStates: result.init_states, climatologyReference: result.climatology_reference, doySincos: result.doy_sincos, landSeaMask: result.land_sea_mask, source: result.source, label: `Seasonal climatology · day ${doy}` };
}

function encodeFloat32(values: number[], shape: number[]): { b64: string; shape: number[] } {
  const buffer = new ArrayBuffer(values.length * 4);
  const view = new DataView(buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  return { b64: btoa(binary), shape };
}

export async function simulate(request: SimulationRequest, signal?: AbortSignal): Promise<SimulationResponse> {
  const body = {
    ...request,
    init_states: encodeFloat32(request.init_states.flat(3), [3, 6, 32, 36]),
    perturbation: request.perturbation ? encodeFloat32(request.perturbation.flat(2), [6, 32, 36]) : null,
  };
  const result = await postJson<SimulationResponse>("/simulate", body, signal);
  if (!Array.isArray(result?.rollout) || result.rollout.length !== request.steps) throw new Error("Simulation response has an unexpected rollout length");
  for (const step of result.rollout) {
    if (!Array.isArray(step) || step.length !== 6 || step.some((channel) => !Array.isArray(channel) || channel.length !== 32 || channel.some((row) => !Array.isArray(row) || row.length !== 36 || row.some((value) => !Number.isFinite(value))))) {
      throw new Error("Simulation response has invalid shape or non-finite values");
    }
    for (const value of step[1].flat()) if (value < -1e-6) throw new Error("Server returned negative precipitation");
    for (const value of step[0].flat()) if (value < 233.15 || value > 333.15) throw new Error("Server returned temperature outside the physical validity gate");
  }
  return result;
}
