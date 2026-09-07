import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

// HTTPS is required for getUserMedia on any host that isn't localhost —
// this is what lets a phone on the same LAN grant camera access.
export default defineConfig({
  plugins: [basicSsl()],
  server: {
    host: true,
  },
  base: "./",
});
