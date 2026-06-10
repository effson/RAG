import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/import-api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/import-api/, '')
      },
      '/query-api': {
        target: 'http://0.0.0.0:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/query-api/, '')
      }
    }
  }
})
