import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_API_URL || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        // The API and the dev server share an origin, so no CORS in development.
        '/api': { target: apiTarget, changeOrigin: true },
        '/health': { target: apiTarget, changeOrigin: true },
      },
    },
  }
})
