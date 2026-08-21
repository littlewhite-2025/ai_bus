import { MorphIcon } from "morphicons/react";
import { Sun, Moon } from "lucide"; // icon data, not components

export default function ThemeToggle({ theme, onToggle }) {
  const isLight = theme === "light";

  return (
    <button
      type="button"
      className="icon-btn"
      onClick={onToggle}
      aria-pressed={isLight}
      aria-label={isLight ? "切換為深色模式" : "切換為淺色模式"}
      title={isLight ? "切換為深色模式" : "切換為淺色模式"}
    >
      <MorphIcon icon={isLight ? Sun : Moon} size={18} spring="smooth" />
    </button>
  );
}
