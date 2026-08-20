import { ROUTE_COLORS, crowdLevel } from "./format.js";

export function renderBusHeader(container, bus) {
  const c = ROUTE_COLORS[bus.routeColor];
  const { occupied_count, total_seats } = bus.seats;
  const empty = total_seats - occupied_count;
  const level = crowdLevel(occupied_count, total_seats);
  const pct = total_seats ? Math.round((occupied_count / total_seats) * 100) : 0;

  container.style.setProperty("--route-accent", c.accent);
  container.style.setProperty("--route-accent-dim", c.dim);

  container.innerHTML = `
    <div class="bus-card__row">
      <div>
        <div class="route-badge">${bus.route}<span class="route-badge__label">${c.label}公車</span></div>
      </div>
      <div class="bus-card__plate tabular">${bus.plate}</div>
    </div>

    <div class="bus-card__figure">
      <div>
        <div class="bus-card__number tabular">${empty}<span class="bus-card__number-unit">席</span></div>
        <div class="bus-card__number-label">目前空位</div>
      </div>

      <div class="crowd-meter">
        <div class="crowd-meter__label">
          <span>擁擠程度</span>
          <span class="crowd-meter__value" style="color:${level.color}">${level.label}</span>
        </div>
        <div class="crowd-meter__track">
          <div class="crowd-meter__fill" style="--crowd-pct:${pct}%; --crowd-color:${level.color}"></div>
        </div>
      </div>
    </div>
  `;
}
