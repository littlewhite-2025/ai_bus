import { useMemo, useState } from "react";
import SearchField from "./SearchField.jsx";
import RouteChip from "./RouteChip.jsx";

export default function Sidebar({ fleet, activeId, onSelect }) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return fleet;
    return fleet.filter(
      (b) => b.route.toLowerCase().includes(q) || b.plate.toLowerCase().includes(q)
    );
  }, [fleet, query]);

  return (
    <aside className="sidebar">
      <div className="sidebar__eyebrow">路線</div>
      <div className="sidebar__search">
        <SearchField value={query} onChange={setQuery} />
      </div>
      <ul className="route-list">
        {filtered.map((bus) => (
          <RouteChip key={bus.id} bus={bus} isActive={bus.id === activeId} onSelect={onSelect} />
        ))}
      </ul>
    </aside>
  );
}
