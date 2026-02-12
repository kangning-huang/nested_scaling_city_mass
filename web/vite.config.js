import { defineConfig } from 'vite'

export default defineConfig({
  // Use relative paths so the app works under GitHub Pages project paths.
  base: './',
  server: { port: 5173 },
})
