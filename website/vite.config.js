import { defineConfig } from 'vite';
import { renderMath } from './scripts/render-math.mjs';

export default defineConfig({
  base: './',
  plugins: [{
    name: 'prerender-paper-math',
    transformIndexHtml: { order: 'pre', handler: renderMath },
  }],
});
