import { connectionStatusOf, formatTime } from "../utils/format";

export default function ConsoleStrip({ bus }) {
  const status = connectionStatusOf(bus.connectionStatus);

  return (
    <footer className="console" role="status">
      <span className="console__status" style={{ color: status.color }}>
        <span
          className={`live-dot${status.live ? " live-dot--connected" : ""}`}
          style={{ background: status.color }}
        />
        {status.label}
      </span>
      <span className="console__timestamp tabular">AI 即時更新時間 {formatTime(bus.updatedAt)}</span>
    </footer>
  );
}
