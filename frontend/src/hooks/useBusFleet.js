import { useCallback, useEffect, useRef, useState } from "react";
import { fetchFleet, fetchBus } from "../api/busApi";
import { DEFAULT_FLEET } from "../constants/defaults";

const POLL_MS = 7000;

export function useBusFleet() {
  const [fleet, setFleet] = useState(DEFAULT_FLEET);
  const [activeId, setActiveId] = useState(DEFAULT_FLEET[0]?.id ?? null);
  const [activeBus, setActiveBus] = useState(DEFAULT_FLEET[0] ?? null);
  const pollRef = useRef(null);

  const refreshActive = useCallback(async (id) => {
    if (!id) return;
    const bus = await fetchBus(id);
    if (!bus) return;
    setActiveBus(bus);
    setFleet((prev) => prev.map((b) => (b.id === id ? bus : b)));
  }, []);

  // Initial fleet load (route directory + whatever the backend currently
  // reports, or the default state if it isn't configured yet).
  useEffect(() => {
    let cancelled = false;
    fetchFleet().then((data) => {
      if (cancelled || !data?.length) return;
      setFleet(data);
      setActiveId((prev) => (data.some((b) => b.id === prev) ? prev : data[0].id));
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Poll the selected bus for a fresh snapshot.
  useEffect(() => {
    if (!activeId) return undefined;
    refreshActive(activeId);
    pollRef.current = window.setInterval(() => refreshActive(activeId), POLL_MS);
    return () => window.clearInterval(pollRef.current);
  }, [activeId, refreshActive]);

  return { fleet, activeId, activeBus, selectBus: setActiveId };
}
