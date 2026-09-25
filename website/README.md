# Dr. MAS Project Page

Project page for **Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM
Systems**, built with HTML, CSS, JavaScript, Vite, and build-time KaTeX.

## Local Preview

Requires Node.js 22.12+.

```bash
cd website
npm ci
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

`dist/` is the deployable static site. Relative asset paths support GitHub Pages
at <https://langfengq.github.io/DrMAS/>. Pushing this review branch does not deploy it.

Edit page content in `index.html` and `src/`. Display equations use `$$...$$`
inside `data-math` elements and are rendered during the build.

The supplied manuscript is kept once at [public/paper.pdf](public/paper.pdf).
Figures are from the paper; fonts and their licenses are bundled locally.
