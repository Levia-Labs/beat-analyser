import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// task run serves the API on :5000. The web app refuses POSTs whose Origin is another
// host, and the dev server's origin is localhost:5173, so the proxy presents Flask's own.
const API = "http://127.0.0.1:5000";
const PROXIED = [
  "/upload",
  "/download_yt",
  "/folders",
  "/download",
  "/projects",
  "/config",
  "/login",
  "/logout",
];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      PROXIED.map((path) => [path, { target: API, changeOrigin: true, headers: { origin: API } }]),
    ),
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
