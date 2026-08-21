import { useState } from "react";
import BusCard from "./BusCard.jsx";
import StatRow from "./StatRow.jsx";
import ToggleButton from "./ToggleButton.jsx";
import SeatPanel from "./SeatPanel.jsx";
import ConsoleStrip from "./ConsoleStrip.jsx";

export default function HomePage({ seats, updatedAt, connectionStatus }) {
  const [seatPanelOpen, setSeatPanelOpen] = useState(false);

  return (
    <>
      <div className="content-grid">
        <BusCard seats={seats} />
        <StatRow seats={seats} />

        <ToggleButton isOpen={seatPanelOpen} onToggle={() => setSeatPanelOpen((v) => !v)} />
        <SeatPanel seats={seats} isOpen={seatPanelOpen} />
      </div>

      <ConsoleStrip connectionStatus={connectionStatus} updatedAt={updatedAt} />
    </>
  );
}
