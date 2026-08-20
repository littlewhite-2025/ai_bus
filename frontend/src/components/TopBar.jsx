import { BusIcon } from "../utils/icons.jsx";
import { connectionStatusOf } from "../utils/format";

export default function TopBar({ connectionStatus }) {
  const status = connectionStatusOf(connectionStatus);

  return (
    <header className="topbar">
      <div className="topbar__brand">
        <span className="brand-mark">
          <BusIcon />
        </span>
        <div>
          <div className="topbar__title">AI 智慧公車</div>
          <div className="topbar__subtitle">即時座位監控系統</div>
        </div>
      </div>

      <span className="live-pill">
        <span
          className={`live-dot${status.live ? " live-dot--connected" : ""}`}
          style={{ background: status.color }}
        />
        {status.label}
      </span>
    </header>
  );
}
