import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

// HTTPS is required for getUserMedia on any host that isn't localhost —
// this is what lets a phone on the same LAN grant camera access.
export default defineConfig(({ command }) => ({
  plugins: [basicSsl()],
  server: {
    host: true,
  },
  // Dev serves from the domain root; the GitHub Pages build is a project
  // page under /naruto-ar-jutsu/. Runtime code reads this back via
  // import.meta.env.BASE_URL rather than hardcoding either path.
  base: command === "build" ? "/naruto-ar-jutsu/" : "/",
}));
