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

export function GithubIcon(props) {
  // Lucide deliberately excludes brand marks, so this is a hand-drawn
  // Octocat glyph rather than sourced from the icon-data package.
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M12 2C6.48 2 2 6.58 2 12.2c0 4.49 2.87 8.3 6.84 9.64.5.1.68-.22.68-.48 0-.24-.01-.87-.01-1.71-2.78.62-3.37-1.36-3.37-1.36-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1 .07 1.53 1.05 1.53 1.05.89 1.56 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.31.1-2.73 0 0 .84-.28 2.75 1.05a9.29 9.29 0 0 1 5.01 0c1.91-1.33 2.75-1.05 2.75-1.05.55 1.42.2 2.47.1 2.73.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.8-4.57 5.06.36.32.68.95.68 1.92 0 1.39-.01 2.51-.01 2.85 0 .27.18.59.69.48A10.02 10.02 0 0 0 22 12.2C22 6.58 17.52 2 12 2Z" />
    </svg>
  );
}

export function ArrowUpIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M12 19V5" />
      <path d="m6 11 6-6 6 6" />
    </svg>
  );
}
