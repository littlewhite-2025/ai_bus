import { buildSeatMatrix } from "../utils/format";

export default function SeatPanel({ bus, isOpen }) {
  const matrix = buildSeatMatrix(bus.seats.occupied_seats);
  const empty = bus.seats.total_seats - bus.seats.occupied_count;

  return (
    <section className={`seat-panel${isOpen ? " is-open" : ""}`} aria-label="車內座位圖">
      <div className="seat-panel__inner">
        <div className="seat-panel__head">
          <span className="seat-panel__title">車內座位</span>
          <div className="legend">
            <span className="legend__item">
              <span className="legend__dot" style={{ background: "var(--status-empty)" }} />
              空位
            </span>
            <span className="legend__item">
              <span className="legend__dot" style={{ background: "var(--status-occupied)" }} />
              已坐人
            </span>
          </div>
        </div>

        <div className="bus-schematic">
          <span className="bus-schematic__aisle" />
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-3)" }}>
            {matrix.map((row, i) => (
              <div className="seat-grid" style={{ "--seat-cols": 4 }} key={i}>
                {row.map((seat) => (
                  <div
                    className="seat"
                    data-state={seat.state}
                    key={seat.id}
                    title={`${seat.id} · ${seat.state === "occupied" ? "已坐人" : "空位"}`}
                  >
                    <span className="seat__id">{seat.id}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        <div className="seat-panel__foot">
          <span>剩餘座位 {empty} / {bus.seats.total_seats} 席</span>
          <span>座位以 AI 影像即時偵測</span>
        </div>
      </div>
    </section>
  );
}
