import { ChevronIcon } from "../utils/icons.jsx";

export default function ToggleButton({ isOpen, onToggle }) {
  return (
    <button
      type="button"
      className="cta"
      aria-expanded={isOpen}
      aria-controls="seat-panel"
      onClick={onToggle}
    >
      <span>{isOpen ? "隱藏車內座位" : "查看車內座位"}</span>
      <ChevronIcon />
    </button>
  );
}
