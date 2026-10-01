# Docs site: setup

Copy everything in this folder into the root of the `main` branch:

```text
mkdocs.yml                  site config and navigation (38 pages)
requirements-docs.txt       mkdocs + pymdown-extensions (pinned)
docs/                       the Markdown pages; diagrams and result charts in docs/_snippets/
docs_theme/                 the custom theme (main.html, docs.css, docs.js),
                            make_diagrams.py, make_results.py, data/results.json
                            and AUTHORING.md (writing guide)
.github/workflows/docs.yml  builds on push to main, publishes to gh-pages:/docs
.devin/wiki.json            steers what DeepWiki writes
```

Preview locally:

```bash
pip install -r requirements-docs.txt
mkdocs serve            # http://127.0.0.1:8000, reloads on save
mkdocs build --strict   # fails on broken links or anchors, same as CI
```

After the first push, the workflow publishes the site to
`https://fm8995610-ops.github.io/MAS-PromptBench/docs/`. It replaces only the `docs/`
folder on the `gh-pages` branch, so `index.html` and `quickstart.html` stay as they are.

## DeepWiki

1. Open https://deepwiki.com, submit `https://github.com/fm8995610-ops/MAS-PromptBench` and wait for the first build.
2. Add the badge to the top of `README.md` (the badge also keeps the wiki refreshed):

   ```markdown
   [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fm8995610-ops/MAS-PromptBench)
   ```

3. `.devin/wiki.json` tells DeepWiki which pages to write; regenerate the wiki after changing it.

## Editing

- Add a page: create the `.md` file under `docs/` and add it to `nav` in `mkdocs.yml`.
- Change a topology diagram: edit `docs_theme/make_diagrams.py` and run `python docs_theme/make_diagrams.py`.
- Change or add results: edit `docs_theme/data/results.json` (the paper's published GEPA numbers) and run
  `python docs_theme/make_results.py`. It rewrites every heatmap and bar chart in `docs/_snippets/results/`,
  so the Results Explorer, the task pages and the topology pages stay consistent.
- Follow `docs_theme/AUTHORING.md` for voice, page shape and the Markdown features the theme styles
  (fact strips, card grids, callouts, tabs, code titles).

## Page transition

`docs_theme/assets/mpb-transition.css` and `mpb-transition.js` add the quiet "Short glide, ordered" transition
between the docs and the project page. The project page needs the same two files; see the
`mpb-page-transition` package for both sides.
