# Docs site

Everything for the documentation site lives in this folder:

```text
docs/
├── mkdocs.yml          site config and navigation (38 pages)
├── requirements.txt    mkdocs + pymdown-extensions (pinned)
├── content/            the Markdown pages; diagrams and result charts in content/_snippets/
└── theme/              the custom theme (main.html, docs.css, docs.js), make_diagrams.py,
                        make_results.py, data/results.json and AUTHORING.md (writing guide)
```

Two files sit outside it because their tools look for them at fixed paths:
`.github/workflows/docs.yml` (builds on push to `main` and publishes to `gh-pages:/docs`) and
`.devin/wiki.json` (steers what DeepWiki writes).

Preview from the repository root:

```bash
pip install -r docs/requirements.txt
mkdocs serve -f docs/mkdocs.yml            # http://127.0.0.1:8000, reloads on save
mkdocs build --strict -f docs/mkdocs.yml   # fails on broken links or anchors, same as CI
```

The workflow publishes the site to `https://fm8995610-ops.github.io/MAS-PromptBench/docs/`.
It replaces only the `docs/` folder on the `gh-pages` branch, so `index.html` and `quickstart.html`
stay as they are.

## DeepWiki

1. Open https://deepwiki.com, submit `https://github.com/fm8995610-ops/MAS-PromptBench` and wait for the first build.
2. The badge at the top of the repository `README.md` links to the wiki and keeps it refreshed.
3. `.devin/wiki.json` tells DeepWiki which pages to write; regenerate the wiki after changing it.

## Editing

- Add a page: create the `.md` file under `docs/content/` and add it to `nav` in `docs/mkdocs.yml`.
- Change a topology diagram: edit `docs/theme/make_diagrams.py` and run `python docs/theme/make_diagrams.py`.
- Change or add results: edit `docs/theme/data/results.json` (the paper's published GEPA numbers) and run
  `python docs/theme/make_results.py`. It rewrites every heatmap and bar chart in `docs/content/_snippets/results/`,
  so the Results Explorer, the task pages and the topology pages stay consistent.
- Follow `docs/theme/AUTHORING.md` for voice, page shape and the Markdown features the theme styles
  (fact strips, card grids, callouts, tabs, code titles).

## Page transition

`docs/theme/assets/mpb-transition.css` and `mpb-transition.js` add the "Short glide, ordered" transition
between the docs and the project page. The project page on `gh-pages` loads the same two files from
`static/css/` and `static/js/`.
