import { useCallback, useEffect, useRef, useState } from "react";
import { fetchSeatSnapshot, fallbackSeatSnapshot, ApiNotConfiguredError } from "../api/busApi";

const POLL_MS = 4000;

/**
 * Polls GET /api/bus and exposes { seats, updatedAt, connectionStatus }.
 *
 * connectionStatus:
 *   "idle"      – VITE_API_BASE_URL isn't set; showing the initialized
 *                 default snapshot (all seats empty) until it is.
 *   "connected" – last poll succeeded; seats/updatedAt reflect the response.
 *   "error"     – last poll failed (network/CORS/5xx); seats/updatedAt keep
 *                 the last known-good value rather than resetting to empty.
 */
export function useSeatStatus() {
  const [seats, setSeats] = useState(fallbackSeatSnapshot());
  const [updatedAt, setUpdatedAt] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState("idle");
  const pollRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const data = await fetchSeatSnapshot();
      setSeats(data);
      setUpdatedAt(new Date());
      setConnectionStatus("connected");
    } catch (err) {
      if (err instanceof ApiNotConfiguredError) {
        setConnectionStatus("idle");
        return;
      }
      console.error("[useSeatStatus] fetchSeatSnapshot failed:", err);
      setConnectionStatus("error");
    }
  }, []);

  useEffect(() => {
    refresh();
    pollRef.current = window.setInterval(refresh, POLL_MS);
    return () => window.clearInterval(pollRef.current);
  }, [refresh]);

  return { seats, updatedAt, connectionStatus };
}
