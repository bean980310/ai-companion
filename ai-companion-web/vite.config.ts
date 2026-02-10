import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy Gradio API calls to the backend
      '/api': {
        target: 'http://localhost:7861',
        changeOrigin: true,
      },
      '/gradio_api': {
        target: 'http://localhost:7861',
        changeOrigin: true,
      },
    },
  },
})
