import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    host: '0.0.0.0',
    strictPort: false,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
        rewrite: (path) => path
      },
      '/auth': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
        rewrite: (path) => path
      },
      '/ws': {
        target: 'ws://127.0.0.1:8001',
        ws: true,
      },
      '/detections': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
        rewrite: (path) => path
      },
      '/logout': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
        rewrite: (path) => path
      }
    }
  }
})
