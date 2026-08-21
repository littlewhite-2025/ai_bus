import { BUS_META } from "../constants/defaults";
import { routeColorOf, crowdLevel } from "../utils/format";

export default function BusCard({ seats }) {
  const c = routeColorOf(BUS_META.routeColor);
  const { occupied_count, total_seats } = seats;
  const empty = total_seats - occupied_count;
  const level = crowdLevel(occupied_count, total_seats);
  const pct = total_seats ? Math.round((occupied_count / total_seats) * 100) : 0;

  return (
    <section
      className="bus-card"
      aria-live="polite"
      style={{ "--route-accent": c.accent, "--route-accent-dim": c.dim }}
    >
      <div className="bus-card__row">
        <div className="route-badge">
          {BUS_META.route}
          <span className="route-badge__label">{c.label}公車</span>
        </div>
        <div className="bus-card__plate tabular">{BUS_META.plate}</div>
      </div>

      <div className="bus-card__figure">
        <div>
          <div className="bus-card__number tabular">
            {empty}
            <span className="bus-card__number-unit">席</span>
          </div>
          <div className="bus-card__number-label">目前空位</div>
        </div>

        <div className="crowd-meter">
          <div className="crowd-meter__label">
            <span>擁擠程度</span>
            <span className="crowd-meter__value" style={{ color: level.color }}>
              {level.label}
            </span>
          </div>
          <div className="crowd-meter__track">
            <div
              className="crowd-meter__fill"
              style={{ "--crowd-pct": `${pct}%`, "--crowd-color": level.color }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
