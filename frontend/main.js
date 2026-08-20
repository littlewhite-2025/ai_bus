import { getFleet, getBus, refreshBus } from "./mockApi.js";
import { renderRouteList, renderSearchField } from "./routeList.js";
import { renderBusHeader } from "./busHeader.js";
import { renderStatCards } from "./statCards.js";
import { renderSeatPanel, pulseScan } from "./seatMap.js";
import { renderConsole } from "./console.js";

const els = {
  sidebar: document.getElementById("routeList"),
  search: document.getElementById("searchField"),
  busCard: document.getElementById("busCard"),
  statRow: document.getElementById("statRow"),
  seatPanel: document.getElementById("seatPanel"),
  toggle: document.getElementById("toggleSeatMap"),
  console: document.getElementById("console"),
};

const REFRESH_MS = 7000;
const state = {
  fleet: [],
  activeId: null,
  panelOpen: window.matchMedia("(min-width: 960px)").matches,
};

async function selectBus(id) {
  state.activeId = id;
  const bus = await getBus(id);
  paint(bus);
  renderRouteList(els.sidebar, state.fleet, state.activeId, selectBus);
}

function paint(bus) {
  renderBusHeader(els.busCard, bus);
  renderStatCards(els.statRow, bus.seats);
  renderSeatPanel(els.seatPanel, bus);
  renderConsole(els.console, bus);
  syncPanelState();
}

function syncPanelState() {
  els.seatPanel.classList.toggle("is-open", state.panelOpen);
  els.toggle.setAttribute("aria-expanded", String(state.panelOpen));
  els.toggle.querySelector(".cta__label").textContent = state.panelOpen
    ? "隱藏車內座位"
    : "查看車內座位";
}

async function tick() {
  if (!state.activeId) return;
  const bus = await refreshBus(state.activeId);
  const fresh = await getFleet();
  state.fleet = fresh;
  paint(bus);
  pulseScan(els.seatPanel);
}

function initTheme() {
  const toggle = document.getElementById("themeToggle");
  if (!toggle) return;

  const currentTheme = localStorage.getItem("theme") || "dark";
  document.body.setAttribute("data-theme", currentTheme);

  toggle.addEventListener("click", () => {
    const isLight = document.body.getAttribute("data-theme") === "light";
    const nextTheme = isLight ? "dark" : "light";
    document.body.setAttribute("data-theme", nextTheme);
    localStorage.setItem("theme", nextTheme);
  });
}

function init() {
  initTheme();
  renderSearchField(els.search);
  els.toggle.addEventListener("click", () => {
    state.panelOpen = !state.panelOpen;
    syncPanelState();
  });

  els.search.addEventListener("input", (e) => {
    const target = e.target;
    if (!target.matches("input")) return;
    const q = target.value.trim();
    const filtered = q
      ? state.fleet.filter(
          (b) => b.route.includes(q) || b.plate.toLowerCase().includes(q.toLowerCase())
        )
      : state.fleet;
    renderRouteList(els.sidebar, filtered, state.activeId, selectBus);
  });

  getFleet().then((fleet) => {
    state.fleet = fleet;
    renderRouteList(els.sidebar, fleet, fleet[0].id, selectBus);
    selectBus(fleet[0].id);
  });

  window.setInterval(tick, REFRESH_MS);
}

init();
