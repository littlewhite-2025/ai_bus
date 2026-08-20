import { SeatEmptyIcon, UsersIcon, GaugeIcon } from "../utils/icons.jsx";

export default function StatRow({ seats }) {
  const tiles = [
    {
      key: "empty",
      icon: <SeatEmptyIcon />,
      accent: "var(--status-empty)",
      accentDim: "var(--status-empty-dim)",
      label: "空位",
      value: seats.total_seats - seats.occupied_count,
    },
    {
      key: "occupied",
      icon: <UsersIcon />,
      accent: "var(--status-occupied)",
      accentDim: "var(--status-occupied-dim)",
      label: "已乘坐",
      value: seats.occupied_count,
    },
    {
      key: "person",
      icon: <GaugeIcon />,
      accent: "var(--route-accent)",
      accentDim: "var(--route-accent-dim)",
      label: "車內人數",
      value: seats.person_count,
    },
  ];

  return (
    <section className="stat-row" aria-label="座位統計">
      {tiles.map((t) => (
        <div className="stat-tile" key={t.key}>
          <span className="stat-tile__icon" style={{ color: t.accent, background: t.accentDim }}>
            {t.icon}
          </span>
          <span>
            <span className="stat-tile__value tabular">{t.value}</span>
            <span className="stat-tile__unit">席</span>
          </span>
          <span className="stat-tile__label">{t.label}</span>
        </div>
      ))}
    </section>
  );
}
