import { useState } from "react";
import { useSeatStatus } from "./hooks/useSeatStatus.js";
import TopBar from "./components/TopBar.jsx";
import BusCard from "./components/BusCard.jsx";
import StatRow from "./components/StatRow.jsx";
import ToggleButton from "./components/ToggleButton.jsx";
import SeatPanel from "./components/SeatPanel.jsx";
import ConsoleStrip from "./components/ConsoleStrip.jsx";

export default function App() {
  const { seats, updatedAt, connectionStatus } = useSeatStatus();
  const [seatPanelOpen, setSeatPanelOpen] = useState(false);

  return (
    <div className="app">
      <TopBar connectionStatus={connectionStatus} />

      <main className="main">
        <div className="content-grid">
          <BusCard seats={seats} />
          <StatRow seats={seats} />

          <ToggleButton isOpen={seatPanelOpen} onToggle={() => setSeatPanelOpen((v) => !v)} />
          <SeatPanel seats={seats} isOpen={seatPanelOpen} />
        </div>

        <ConsoleStrip connectionStatus={connectionStatus} updatedAt={updatedAt} />
      </main>
    </div>
  );
}
