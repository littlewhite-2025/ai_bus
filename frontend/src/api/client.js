// Vite only exposes env vars explicitly prefixed with VITE_ to client code,
// so nothing sensitive can leak here by accident.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const REQUEST_TIMEOUT_MS = 8000;

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Thrown specifically when no backend URL has been configured yet. */
export class ApiNotConfiguredError extends ApiError {
  constructor() {
    super("VITE_API_BASE_URL is not set", 0);
    this.name = "ApiNotConfiguredError";
  }
}

/**
 * GET a JSON resource from the inference backend.
 *
 * Kept deliberately narrow: GET-only, same-origin-friendly, no credentials
 * sent by default, and a hard timeout so a stalled request can never hang
 * the UI. Widen this only when a real write endpoint is needed.
 */
export async function apiGet(path, { signal } = {}) {
  if (!API_BASE_URL) {
    throw new ApiNotConfiguredError();
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  if (signal) {
    signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      method: "GET",
      headers: { Accept: "application/json" },
      credentials: "omit",
      signal: controller.signal,
    });

    if (!res.ok) {
      throw new ApiError(`Request failed: ${res.status} ${res.statusText}`, res.status);
    }

    return await res.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new ApiError("Request timed out", 0);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
