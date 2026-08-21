import { useState } from "react";
import { useSeatStatus } from "./hooks/useSeatStatus.js";
import { useTheme } from "./hooks/useTheme.js";
import TopBar from "./components/TopBar.jsx";
import HomePage from "./components/HomePage.jsx";
import AboutPage from "./components/AboutPage.jsx";
import NewsPage from "./components/NewsPage.jsx";
import SiteFooter from "./components/SiteFooter.jsx";
import ScrollTopButton from "./components/ScrollTopButton.jsx";

export default function App() {
  const { seats, updatedAt, connectionStatus } = useSeatStatus();
  const { theme, toggleTheme } = useTheme();
  const [page, setPage] = useState("home");

  return (
    <div className="app">
      <TopBar
        connectionStatus={connectionStatus}
        theme={theme}
        onToggleTheme={toggleTheme}
        currentPage={page}
        onNavigate={setPage}
      />

      <main className="main">
        {page === "about" && <AboutPage />}
        {page === "news" && <NewsPage />}
        {page === "home" && (
          <HomePage seats={seats} updatedAt={updatedAt} connectionStatus={connectionStatus} />
        )}
      </main>

      <SiteFooter />
      <ScrollTopButton />
    </div>
  );
}
