const base = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export function BusIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M4 16V6a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v10" />
      <path d="M4 16h16v2a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-2Z" />
      <path d="M4 11h16" />
    </svg>
  );
}

export function SearchIcon(props) {
  return (
    <svg {...base} strokeWidth={2} {...props}>
      <circle cx="11" cy="11" r="7" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

export function ChevronIcon(props) {
  return (
    <svg {...base} strokeWidth={2} {...props}>
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

export function SeatEmptyIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M6 13V6a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v3" />
      <path d="M6 13h9a2 2 0 0 1 2 2v3H8a2 2 0 0 1-2-2v-3Z" />
      <path d="M17 13V9a2 2 0 0 1 2-2v0a2 2 0 0 1 2 2v9" />
    </svg>
  );
}

export function UsersIcon(props) {
  return (
    <svg {...base} {...props}>
      <circle cx="9" cy="8" r="3" />
      <path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6" />
      <circle cx="17" cy="9" r="2.5" />
      <path d="M15.5 14a4.5 4.5 0 0 1 5.5 4.4" />
    </svg>
  );
}

export function GaugeIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M4 13a8 8 0 0 1 16 0" />
      <path d="M12 13 15 9" />
      <circle cx="12" cy="13" r="1" />
    </svg>
  );
}
