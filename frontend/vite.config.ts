import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import tailwindcss from '@tailwindcss/vite'


// vite.config.ts
export default defineConfig({
  plugins: [react(),tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",  // This should work if your backend listens on 8000
      "/ws": { target: "ws://localhost:8000", ws: true }
    }
  }
});
