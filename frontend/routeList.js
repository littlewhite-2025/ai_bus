import { ROUTE_COLORS, crowdLevel } from "./format.js";
import { icons } from "./icons.js";

export function renderRouteList(container, fleet, activeId, onSelect) {
  container.innerHTML = "";
  fleet.forEach((bus) => {
    const c = ROUTE_COLORS[bus.routeColor];
    const level = crowdLevel(bus.seats.occupied_count, bus.seats.total_seats);
    const isActive = bus.id === activeId;

    const item = document.createElement("li");
    item.style.listStyle = "none";

    const btn = document.createElement("button");
    btn.className = "route-chip";
    btn.style.setProperty("--chip-accent", c.accent);
    btn.style.setProperty("--chip-accent-dim", c.dim);
    btn.setAttribute("aria-pressed", String(isActive));
    btn.innerHTML = `
      <span class="route-chip__badge">${bus.route.replace(/[^0-9]/g, "") || bus.route[0]}</span>
      <span class="route-chip__body">
        <span class="route-chip__route">${bus.route}</span>
        <span class="route-chip__meta">${bus.plate} · ${level.label}</span>
      </span>
      <span class="route-chip__status" style="background:${level.color}"></span>
    `;
    btn.addEventListener("click", () => onSelect(bus.id));
    item.appendChild(btn);
    container.appendChild(item);
  });
}

export function renderSearchField(container) {
  container.innerHTML = `
    <label class="search-field">
      ${icons.search}
      <input type="text" placeholder="搜尋路線或車號" aria-label="搜尋路線或車號" />
    </label>
  `;
}
