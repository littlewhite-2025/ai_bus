// Single-vehicle deployment: the backend's JSON contract only ever outputs
// occupied_seats / occupied_count / total_seats / person_count (see
// run_inference.py's build_result()) — there is no vehicle ID in that
// payload. Rather than inventing one on the frontend, route/plate are just
// fixed display copy here. Swap these two constants directly if the
// physical bus changes; don't wire them up to any kind of lookup.
export const BUS_META = {
  route: "default",
  routeColor: "red",
  plate: "AAA-1234",
};

export const TOTAL_SEATS = 16;

/**
 * Builds a fully-empty occupied_seats map using the same seat-ID scheme as
 * the inference backend:
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

/** Initial state before the first successful response from the backend. */
export function createDefaultSeatSnapshot(total = TOTAL_SEATS) {
  return {
    occupied_seats: createDefaultSeatMap(total),
    occupied_count: 0,
    total_seats: total,
    person_count: 0,
  };
}
