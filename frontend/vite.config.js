import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "0.0.0.0",
    allowedHosts: true,

    proxy: {
      "/api": {
        target: process.env.VITE_PROXY_BACKEND || "http://localhost:8000",
        changeOrigin: true,
      },
      "/healthz": {
        target: process.env.VITE_PROXY_BACKEND || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
