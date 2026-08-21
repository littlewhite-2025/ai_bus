import { useState } from "react";
import { MorphIcon } from "morphicons/react";
import { Menu, X } from "lucide"; // icon data, not components
import { BusIcon, GithubIcon } from "../utils/icons.jsx";
import { connectionStatusOf } from "../utils/format";
import { NAV_LINKS, GITHUB_URL } from "../constants/site";
import ThemeToggle from "./ThemeToggle.jsx";
import NavMenu from "./NavMenu.jsx";

export default function TopBar({ connectionStatus, theme, onToggleTheme, currentPage, onNavigate }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const status = connectionStatusOf(connectionStatus);

  const handleNavigate = (key) => {
    onNavigate(key);
    setMenuOpen(false);
  };

  return (
    <header className="topbar">
      {/* 左側 icon + 文字：目前先用公車 icon 佔位，之後會換成正式的圖示。
          點擊會重新載入頁面。 */}
      <button
        type="button"
        className="topbar__brand"
        onClick={() => window.location.reload()}
        aria-label="重新載入頁面"
        title="重新載入頁面"
      >
        <span className="brand-mark">
          <BusIcon />
        </span>
        <span className="topbar__brand-text">
          <span className="topbar__title">AI 智慧公車</span>
          <span className="topbar__subtitle">即時座位監控系統</span>
        </span>
      </button>

      <nav className="topbar__nav" aria-label="主要導覽">
        {NAV_LINKS.map((link) => (
          <button
            key={link.key}
            type="button"
            className={`nav-link${currentPage === link.key ? " is-active" : ""}`}
            onClick={() => handleNavigate(link.key)}
          >
            {link.label}
          </button>
        ))}
      </nav>

      <div className="topbar__actions">
        <span className="live-pill">
          <span
            className={`live-dot${status.live ? " live-dot--connected" : ""}`}
            style={{ background: status.color }}
          />
          <span className="live-pill__label">{status.label}</span>
        </span>

        <ThemeToggle theme={theme} onToggle={onToggleTheme} />

        <a
          className="icon-btn"
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="在 GitHub 上查看原始碼"
          title="GitHub"
        >
          <GithubIcon width={17} height={17} />
        </a>

        <button
          type="button"
          className="icon-btn"
          onClick={() => setMenuOpen((v) => !v)}
          aria-expanded={menuOpen}
          aria-controls="nav-menu"
          aria-label={menuOpen ? "關閉選單" : "開啟選單"}
        >
          <MorphIcon icon={menuOpen ? X : Menu} size={20} spring="smooth" />
        </button>
      </div>

      <NavMenu isOpen={menuOpen} currentPage={currentPage} onNavigate={handleNavigate} />
    </header>
  );
}
