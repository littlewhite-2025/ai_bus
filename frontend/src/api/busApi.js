import { apiGet, ApiNotConfiguredError } from "./client";
import { createDefaultSeatSnapshot } from "../constants/defaults";

/**
 * Expected backend contract (mirrors run_inference.py's build_result()
 * exactly — no vehicle ID, no extra fields):
 *
 * GET /api/bus
 *   -> {
 *        occupied_seats: { "A01": "empty" | "occupied", ... },
 *        occupied_count: number,
 *        total_seats: number,
 *        person_count: number
 *      }
 */
export async function fetchSeatSnapshot() {
  return apiGet("/api/bus");
}

export function fallbackSeatSnapshot() {
  return createDefaultSeatSnapshot();
}

export { ApiNotConfiguredError };
