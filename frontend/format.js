export const ROUTE_COLORS = {
  red: { accent: "var(--route-red)", dim: "var(--route-red-dim)", label: "紅線" },
  orange: { accent: "var(--route-orange)", dim: "var(--route-orange-dim)", label: "橘線" },
  blue: { accent: "var(--route-blue)", dim: "var(--route-blue-dim)", label: "藍線" },
  green: { accent: "var(--route-green)", dim: "var(--route-green-dim)", label: "綠線" },
};

/**
 * Derive a crowd level (空曠 / 普通 / 擁擠) from the occupied ratio.
 * Thresholds are intentionally simple; tune once real ridership data exists.
 */
export function crowdLevel(occupied, total) {
  if (!total) return { key: "unknown", label: "無資料", color: "var(--ink-tertiary)" };
  const ratio = occupied / total;
  if (ratio < 0.4) return { key: "spacious", label: "空曠", color: "var(--status-empty)" };
  if (ratio < 0.75) return { key: "normal", label: "普通", color: "var(--status-warn)" };
  return { key: "crowded", label: "擁擠", color: "var(--status-occupied)" };
}

export function formatTime(date) {
  const d = date instanceof Date ? date : new Date(date);
  return d.toLocaleTimeString("zh-TW", { hour12: false });
}

export function padSeat(n) {
  return String(n).padStart(2, "0");
}

/**
 * Build an ordered seat matrix (row-major, back of bus first) from the
 * occupied_seats map returned by the inference backend. Mirrors the
 * numbering scheme documented for the seat-detection model:
 *   A07 A08   B07 B08
 *   A05 A06   B05 B06
 *   A03 A04   B03 B04
 *   A01 A02   B01 B02
 */
export function buildSeatMatrix(occupiedSeats) {
  const total = Object.keys(occupiedSeats).length;
  const perBlock = total / 2;
  const rows = Math.max(1, Math.round(perBlock / 2));
  const matrix = [];

  for (let r = rows; r >= 1; r--) {
    const left = (r - 1) * 2 + 1;
    const right = left + 1;
    const row = [
      { id: `A${padSeat(left)}`, block: "A" },
      { id: `A${padSeat(right)}`, block: "A" },
      { id: `B${padSeat(left)}`, block: "B" },
      { id: `B${padSeat(right)}`, block: "B" },
    ].map((s) => ({ ...s, state: occupiedSeats[s.id] ?? "empty" }));
    matrix.push(row);
  }
  return matrix;
}
