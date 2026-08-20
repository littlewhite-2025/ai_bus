import { apiGet, ApiNotConfiguredError } from "./client";
import { ROUTE_DIRECTORY, DEFAULT_FLEET, createDefaultBus } from "../constants/defaults";

/**
 * Expected backend contract (mirrors run_inference.py's output):
 *
 * GET /api/fleet
 *   -> [{ id, route, routeColor, plate, seats: <snapshot>, updatedAt, connectionStatus }, ...]
 *
 * GET /api/buses/:id
 *   -> { id, route, routeColor, plate, seats: <snapshot>, updatedAt, connectionStatus }
 *
 * <snapshot> = { occupied_seats, occupied_count, total_seats, person_count }
 */

export async function fetchFleet() {
  try {
    return await apiGet("/api/fleet");
  } catch (err) {
    if (err instanceof ApiNotConfiguredError) return DEFAULT_FLEET;
    console.error("[busApi] fetchFleet failed, showing default state:", err);
    return DEFAULT_FLEET;
  }
}

export async function fetchBus(busId) {
  const fallback = () => {
    const meta = ROUTE_DIRECTORY.find((b) => b.id === busId) ?? null;
    return meta ? createDefaultBus(meta) : null;
  };

  try {
    return await apiGet(`/api/buses/${encodeURIComponent(busId)}`);
  } catch (err) {
    if (!(err instanceof ApiNotConfiguredError)) {
      console.error(`[busApi] fetchBus(${busId}) failed, showing default state:`, err);
    }
    return fallback();
  }
}
