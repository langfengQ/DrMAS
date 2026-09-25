# Dr. MAS Project Page

A responsive, self-contained academic project page for **Dr. MAS: Stable
Reinforcement Learning for Multi-Agent LLM Systems**.

- Code repository: <https://github.com/langfengQ/DrMAS>
- Intended GitHub Pages URL: <https://langfengq.github.io/DrMAS/>
- Deployment: feature branches are for review only. Publishing requires a merge
  into `master` or `main`, with GitHub Pages enabled for GitHub Actions.

## Visual Review

The review branch includes browser-rendered previews of the final paper-matched
palette: [desktop homepage](review/desktop.png), [mobile homepage](review/mobile.png),
and [interactive results](review/results.png). These review images are not bundled
into the published site.

The deployment workflow is intentionally provided as a template at
`deployment/pages.yml`, not installed in `.github/workflows/`. The current GitHub
token grants `repo` but not `workflow`; this branch therefore does not change
active Actions workflows or publish a live website.

## Local Development

Requires Node.js 22.12+ (Node 24 is also supported).

```bash
npm ci
npm run dev
```

The development server is available on port 5173. The site uses plain HTML, CSS,
and JavaScript with Vite for development and production asset bundling. No backend,
API key, tracking script, external font request, or runtime CDN is required.

```bash
npm test
npm run build
npx playwright install chromium
npm run test:e2e
npm run preview
```

`dist/` is the deployable static site. End-to-end tests serve that artifact at
`/DrMAS/`, not at the domain root, to catch GitHub Pages subpath issues. They cover
all 16 result configurations, asset loading, responsive overflow, the figure
dialog, clipboard success/failure, mobile navigation, and no-JavaScript access.

## Same-Repository GitHub Pages

Keep the training code in the existing `langfengQ/DrMAS` repository. The recommended
layout is:

```text
DrMAS/
  .github/workflows/pages.yml
  website/
    index.html
    package.json
    package-lock.json
    vite.config.js
    playwright.config.js
    Dr__MAS_final.pdf
    src/
    public/
    scripts/
    tests/
    README.md
    .gitignore
  ... existing training code, unchanged
```

After review, install `website/deployment/pages.yml` as
`.github/workflows/pages.yml` at the **repository root** using GitHub's web editor
or a credential with `workflow` permission. It detects the `website/` directory automatically and also works
when this site is the entire repository. It supports the existing `master` default
branch as well as `main`.

1. Review the `website/` source on the feature branch and install the workflow
   template at the repository root. Keep `node_modules/`, `dist/`, and
   `test-results/` out of version control.
2. In `langfengQ/DrMAS`, open **Settings > Pages > Build and deployment** and set
   **Source** to **GitHub Actions**.
3. Merge the website changes, or manually run **Deploy Dr. MAS project page** from
   the Actions tab. The workflow tests, builds, and publishes only `dist/`.
4. After deployment succeeds, the project page is available at
   <https://langfengq.github.io/DrMAS/>. Set that URL as the repository's About
   website and add a Project Page link to the repository README.

Vite uses `base: './'`; asset and PDF links work under `/DrMAS/` without a custom
domain. Canonical and scholarly PDF URLs in `index.html` already use the intended
address. If the repository name or domain changes, update those URLs as well.

## Content Sources And Editing

- `index.html`: title, authors, abstract, method, efficiency study, citation, and
  metadata. Author names and affiliations come from the supplied PDF.
- `src/results.js`: scores transcribed from Tables 1 and 2, including reported
  average rows. Every score pair is `[avg@16, pass@16]`. Negative and zero changes
  are preserved. Displayed gains are differences of the rounded table scores in
  **percentage points**, distinct from the relative improvements in the abstract.
- `src/styles.css`: layout, typography, colors, responsive behavior, and reduced
  motion support.
- `public/paper.pdf`: exact copy of `Dr__MAS_final.pdf`.
- `public/figures/`: original Figures 2 and 3 cropped from the PDF, not synthetic
  or reconstructed experimental plots.
- `public/citation.bib`: downloadable citation; keep it synchronized with the
  visible citation in `index.html`. A test checks they match.
- `public/fonts/`: self-hosted Space Grotesk, DM Sans, and IBM Plex Mono from Google
  Fonts, distributed with their SIL Open Font License files.

The arXiv identifier `2602.08847` and Hugging Face model collection are taken from
the existing repository README. The BibTeX uses that arXiv identifier and the
author list in the supplied manuscript. No unverified DOI or conference acceptance
badge is displayed.

The visual palette follows the paper: Dr. MAS purple `#8c569b` and GRPO orange
`#f7b06f` are taken directly from Figure 3's vector colors. Highlighted result
cells use the original Tables 1-2 background `#eef0f2`. Soft lavender surfaces
and a dark plum method section extend that palette without altering the figures.

To regenerate paper assets after replacing the manuscript:

```bash
python3 -m pip install pymupdf
python3 scripts/extract_figures.py
```

The extraction script uses fixed page numbers and crop coordinates from the
supplied 34-page PDF. If its layout changes, adjust the bounds and visually inspect
the extracted figures. PyMuPDF is needed only to regenerate assets, not to build
or deploy the site.

## Research Scope

The page retains the paper's evaluation limitations. Headline performance gains
are paper-reported relative improvements over multi-agent GRPO. The efficiency
comparison is inference with heterogeneous model assignments; the cost reduction
is an estimate based on OpenRouter prices, not measured GPU training cost.
