import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const proxy = { "/api": process.env.CLS_API_URL || "http://127.0.0.1:8000" };
export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (
            id.includes("node_modules/zrender") ||
            id.includes("node_modules/echarts")
          )
            return "charts";
        },
      },
    },
  },
});
