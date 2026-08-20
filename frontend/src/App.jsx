import { useState } from "react";
import { useBusFleet } from "./hooks/useBusFleet.js";
import TopBar from "./components/TopBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import BusCard from "./components/BusCard.jsx";
import StatRow from "./components/StatRow.jsx";
import ToggleButton from "./components/ToggleButton.jsx";
import SeatPanel from "./components/SeatPanel.jsx";
import ConsoleStrip from "./components/ConsoleStrip.jsx";

export default function App() {
  const { fleet, activeId, activeBus, selectBus } = useBusFleet();
  const [seatPanelOpen, setSeatPanelOpen] = useState(false);

  if (!activeBus) return null;

  return (
    <div className="app">
      <TopBar connectionStatus={activeBus.connectionStatus} />

      <div className="layout">
        <Sidebar fleet={fleet} activeId={activeId} onSelect={selectBus} />

        <main className="main">
          <div className="content-grid">
            <BusCard bus={activeBus} />
            <StatRow seats={activeBus.seats} />

            <ToggleButton isOpen={seatPanelOpen} onToggle={() => setSeatPanelOpen((v) => !v)} />
            <SeatPanel bus={activeBus} isOpen={seatPanelOpen} />
          </div>

          <ConsoleStrip bus={activeBus} />
        </main>
      </div>
    </div>
  );
}
