import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Single-page React app. In dev, /api (and /api/assets) is proxied to the
// FastAPI backend so the browser stays same-origin (no CORS) and image src
// URLs like /api/assets/... resolve directly.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Use 127.0.0.1 (not localhost) so we hit uvicorn's IPv4 bind and avoid
      // ECONNREFUSED on ::1 when localhost resolves to IPv6 on Windows.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
