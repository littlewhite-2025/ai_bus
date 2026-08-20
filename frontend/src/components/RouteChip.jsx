import { routeColorOf, crowdLevel, connectionStatusOf } from "../utils/format";

export default function RouteChip({ bus, isActive, onSelect }) {
  const c = routeColorOf(bus.routeColor);
  const connected = bus.connectionStatus === "connected";
  const level = connected ? crowdLevel(bus.seats.occupied_count, bus.seats.total_seats) : null;
  const status = connectionStatusOf(bus.connectionStatus);
  const badgeNumber = bus.route.replace(/[^0-9]/g, "") || bus.route[0];

  return (
    <li style={{ listStyle: "none" }}>
      <button
        type="button"
        className="route-chip"
        style={{ "--chip-accent": c.accent, "--chip-accent-dim": c.dim }}
        aria-pressed={isActive}
        onClick={() => onSelect(bus.id)}
      >
        <span className="route-chip__badge">{badgeNumber}</span>
        <span className="route-chip__body">
          <span className="route-chip__route">{bus.route}</span>
          <span className="route-chip__meta">
            {bus.plate} · {connected ? level.label : status.label}
          </span>
        </span>
        <span
          className="route-chip__status"
          style={{ background: connected ? level.color : status.color }}
        />
      </button>
    </li>
  );
}
