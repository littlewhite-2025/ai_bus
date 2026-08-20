import { buildSeatMatrix } from "./format.js";

export function renderSeatPanel(container, bus) {
  const matrix = buildSeatMatrix(bus.seats.occupied_seats);
  const empty = bus.seats.total_seats - bus.seats.occupied_count;

  const rows = matrix
    .map(
      (row) => `
      <div class="seat-grid" style="--seat-cols:4">
        ${row
          .map(
            (seat) => `
          <div class="seat" data-state="${seat.state}" title="${seat.id} · ${seat.state === "occupied" ? "已坐人" : "空位"
              }">
            <span class="seat__id">${seat.id}</span>
          </div>`
          )
          .join("")}
      </div>`
    )
    .join("");

  container.innerHTML = `
    <div class="seat-panel__inner">
      <div class="seat-panel__head">
        <span class="seat-panel__title">車內座位</span>
        <div class="legend">
          <span class="legend__item"><span class="legend__dot" style="background:var(--status-empty)"></span>空位</span>
          <span class="legend__item"><span class="legend__dot" style="background:var(--status-occupied)"></span>已坐人</span>
        </div>
      </div>

      <div class="bus-schematic">
        <span class="bus-schematic__scan"></span>
        <span class="bus-schematic__aisle"></span>
        <div class="seat-rows" style="display:flex; flex-direction:column; gap:var(--sp-3)">
          ${rows}
        </div>
      </div>

      <div class="seat-panel__foot">
        <span>剩餘座位 ${empty} / ${bus.seats.total_seats} 席</span>
        <span>座位以 AI 影像即時偵測</span>
      </div>
    </div>
  `;
}

/** Re-triggers the scan-sweep CSS animation after a live data refresh. */
export function pulseScan(container) {
  const scan = container.querySelector(".bus-schematic__scan");
  if (!scan) return;
  scan.style.animation = "none";
  // force reflow so the animation can restart
  void scan.offsetWidth;
  scan.style.animation = "";
}
