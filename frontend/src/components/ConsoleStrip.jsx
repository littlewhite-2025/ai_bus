import { connectionStatusOf, formatTime } from "../utils/format";

export default function ConsoleStrip({ connectionStatus, updatedAt }) {
  const status = connectionStatusOf(connectionStatus);

  return (
    <footer className="console" role="status">
      <span className="console__status" style={{ color: status.color }}>
        <span
          className={`live-dot${status.live ? " live-dot--connected" : ""}`}
          style={{ background: status.color }}
        />
        {status.label}
      </span>
      <span className="console__timestamp tabular">AI 即時更新時間 {formatTime(updatedAt)}</span>
    </footer>
  );
}
