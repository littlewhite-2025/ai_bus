import { formatTime } from "./format.js";

export function renderConsole(container, bus) {
  const ok = bus.connectionStatus === "connected";
  container.innerHTML = `
    <span class="console__status" style="color:${ok ? "var(--status-empty)" : "var(--status-occupied)"}">
      <span class="live-dot" style="background:${ok ? "var(--status-empty)" : "var(--status-occupied)"}"></span>
      ${ok ? "系統連線正常" : "連線中斷，重新連線中"}
    </span>
    <span class="console__timestamp tabular">AI 即時更新時間 ${formatTime(bus.updatedAt)}</span>
  `;
}
