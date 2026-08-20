import { icons } from "./icons.js";

const TILES = [
  {
    key: "empty",
    icon: icons.seatEmpty,
    accent: "var(--status-empty)",
    accentDim: "var(--status-empty-dim)",
    label: "空位",
    value: (s) => s.total_seats - s.occupied_count,
  },
  {
    key: "occupied",
    icon: icons.users,
    accent: "var(--status-occupied)",
    accentDim: "var(--status-occupied-dim)",
    label: "已乘坐",
    value: (s) => s.occupied_count,
  },
  {
    key: "person",
    icon: icons.gauge,
    accent: "var(--route-accent)",
    accentDim: "var(--route-accent-dim)",
    label: "車內人數",
    value: (s) => s.person_count,
  },
];

export function renderStatCards(container, seats) {
  container.innerHTML = TILES.map(
    (t) => `
    <div class="stat-tile">
      <span class="stat-tile__icon" style="color:${t.accent}; background:${t.accentDim}">${t.icon}</span>
      <span>
        <span class="stat-tile__value tabular">${t.value(seats)}</span>
        <span class="stat-tile__unit">席</span>
      </span>
      <span class="stat-tile__label">${t.label}</span>
    </div>`
  ).join("");
}
