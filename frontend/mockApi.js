/**
 * Data layer seam.
 *
 * Every function here returns data shaped exactly like the real backend
 * will (see run_inference.py's JSON contract: occupied_seats /
 * occupied_count / total_seats / person_count). When the Python service is
 * ready, swap the bodies of these two functions for `fetch()` calls —
 * nothing in js/components or js/main.js needs to change.
 */

const TOTAL_SEATS = 16;

function makeSeatIds() {
  const ids = [];
  for (const block of ["A", "B"]) {
    for (let n = 1; n <= TOTAL_SEATS / 2; n++) {
      ids.push(`${block}${String(n).padStart(2, "0")}`);
    }
  }
  return ids;
}

const SEAT_IDS = makeSeatIds();

function randomSeatState(occupiedCount) {
  const occupied = new Set();
  const pool = [...SEAT_IDS];
  while (occupied.size < occupiedCount && pool.length) {
    const idx = Math.floor(Math.random() * pool.length);
    occupied.add(pool.splice(idx, 1)[0]);
  }
  const occupied_seats = {};
  SEAT_IDS.forEach((id) => {
    occupied_seats[id] = occupied.has(id) ? "occupied" : "empty";
  });
  return occupied_seats;
}

function inferenceSnapshot(occupiedCount, personCount) {
  const occupied_seats = randomSeatState(occupiedCount);
  return {
    occupied_seats,
    occupied_count: occupiedCount,
    total_seats: TOTAL_SEATS,
    person_count: personCount ?? occupiedCount,
  };
}

const FLEET = [
  { id: "kka1234", route: "紅28", routeColor: "red", plate: "KKA-1234", baseOccupied: 4 },
  { id: "khh0721", route: "橘1", routeColor: "orange", plate: "KHH-0721", baseOccupied: 11 },
  { id: "kbb5588", route: "藍25", routeColor: "blue", plate: "KBB-5588", baseOccupied: 1 },
  { id: "kgg3302", route: "綠12", routeColor: "green", plate: "KGG-3302", baseOccupied: 8 },
];

const snapshots = new Map(
  FLEET.map((bus) => [bus.id, inferenceSnapshot(bus.baseOccupied)])
);

export async function getFleet() {
  return FLEET.map((bus) => ({
    ...bus,
    seats: snapshots.get(bus.id),
    updatedAt: new Date().toISOString(),
    connectionStatus: "connected",
  }));
}

export async function getBus(id) {
  const bus = FLEET.find((b) => b.id === id);
  if (!bus) return null;
  return {
    ...bus,
    seats: snapshots.get(id),
    updatedAt: new Date().toISOString(),
    connectionStatus: "connected",
  };
}

/** Simulates the AI model re-scanning the cabin and producing a new frame. */
export async function refreshBus(id) {
  const bus = FLEET.find((b) => b.id === id);
  if (!bus) return null;
  const drift = Math.floor(Math.random() * 3) - 1;
  const next = Math.min(TOTAL_SEATS, Math.max(0, bus.baseOccupied + drift));
  bus.baseOccupied = next;
  snapshots.set(id, inferenceSnapshot(next));
  return getBus(id);
}
