// Static route directory. Which buses exist is fleet/config data — it does
// not depend on the AI inference backend, so it is safe to hard-code here
// until there's a real fleet-management endpoint to source it from.
export const ROUTE_DIRECTORY = [
  { id: "kka1234", route: "紅28", routeColor: "red", plate: "KKA-1234" },
  { id: "khh0721", route: "橘1", routeColor: "orange", plate: "KHH-0721" },
  { id: "kbb5588", route: "藍25", routeColor: "blue", plate: "KBB-5588" },
  { id: "kgg3302", route: "綠12", routeColor: "green", plate: "KGG-3302" },
];

export const TOTAL_SEATS = 16;

/**
 * Builds a fully-empty occupied_seats map using the same seat-ID scheme as
 * the inference backend (see run_inference.py's JSON contract):
 *   A01..A08 (left block), B01..B08 (right block)
 */
export function createDefaultSeatMap(total = TOTAL_SEATS) {
  const perBlock = total / 2;
  const map = {};
  ["A", "B"].forEach((block) => {
    for (let n = 1; n <= perBlock; n++) {
      map[`${block}${String(n).padStart(2, "0")}`] = "empty";
    }
  });
  return map;
}

export function createDefaultSeatSnapshot(total = TOTAL_SEATS) {
  return {
    occupied_seats: createDefaultSeatMap(total),
    occupied_count: 0,
    total_seats: total,
    person_count: 0,
  };
}

/**
 * A bus that has not (yet) received a live snapshot from the backend.
 * connectionStatus stays "idle" until fetchBus/fetchFleet succeeds, so the
 * UI can tell "genuinely zero occupancy" apart from "no data yet" if that
 * distinction is ever needed.
 */
export function createDefaultBus(meta) {
  return {
    ...meta,
    seats: createDefaultSeatSnapshot(),
    updatedAt: null,
    connectionStatus: "idle",
  };
}

export const DEFAULT_FLEET = ROUTE_DIRECTORY.map(createDefaultBus);
