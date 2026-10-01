"""Generate the results components used across the docs from docs_theme/data/results.json.

Every number comes from the paper's published GEPA results (also shown on the project
page). Components are plain HTML that the theme styles through CSS tokens, so they work
in light and dark mode and need no JavaScript (a hover tooltip is a progressive extra).

    python docs_theme/make_results.py

Writes docs/_snippets/results/*.html, which pages include with
--8<-- "results/<name>.html".
"""
import json
from html import escape
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "docs_theme" / "data" / "results.json").read_text())
OUT = ROOT / "docs" / "_snippets" / "results"

TASKS = DATA["tasks"]
TASK = {t["id"]: t for t in TASKS}
TOPOS = [t["id"] for t in DATA["topologies"]]
TLABEL = {t["id"]: t["label"] for t in DATA["topologies"]}
FW = DATA["framework_study"]["frameworks"]
STUDIES = {"topology": DATA["topology_study"], "framework": DATA["framework_study"]["cells"]}
DOMAINS = ["Reasoning", "Coding", "Tool-Calling"]


def r1(x):
    """Round half away from zero to one decimal, as the paper reports (4.25 -> 4.3, -1.25 -> -1.3)."""
    q = Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return float(q)


def delta(cell):
    return r1(cell[1] - cell[0])


def fmt_d(d):
    if d == 0:
        return "0.0"
    return ("+" if d > 0 else "−") + f"{abs(d):.1f}"


def step(d):
    """Diverging bucket: z for no change, g1..g4 for gains, l1..l4 for regressions."""
    if d == 0:
        return "z"
    a = abs(d)
    k = 1 if a <= 2 else 2 if a <= 5 else 3 if a <= 10 else 4
    return ("g" if d > 0 else "l") + str(k)


def sc(x):
    return f"{x:.1f}"


def tip(*parts):
    return escape(" · ".join(p for p in parts if p), quote=True)


def cell_html(c, tip_text, cls=""):
    d = delta(c)
    return (f'<td class="dc d-{step(d)}{" " + cls if cls else ""}" data-tip="{tip_text}">'
            f'<b>{fmt_d(d)}</b><span>{sc(c[0])} → {sc(c[1])}</span></td>')


def avg_cell(values, tip_text, cls="avg"):
    """values: list of [base, opt]; shows the mean delta and mean scores."""
    b = mean(v[0] for v in values)
    o = mean(v[1] for v in values)
    d = r1(mean(delta(v) for v in values))
    return (f'<td class="dc d-{step(d)} {cls}" data-tip="{tip_text}">'
            f'<b>{fmt_d(d)}</b><span>{b:.1f} → {o:.1f}</span></td>')


def legend(scores=True):
    keys = [("l4", "−10+"), ("l3", "−5"), ("l2", "−2"), ("l1", ""), ("z", "0"), ("g1", ""), ("g2", "+2"), ("g3", "+5"), ("g4", "+10+")]
    sw = "".join(f'<i class="d-{k}"></i>' for k, _ in keys)
    return ('<div class="dlegend" aria-hidden="true"><span>regression</span>'
            f'<span class="dlegend-scale">{sw}</span><span>gain</span>'
            f'<span class="dlegend-note">{"Δ in points · cell shows baseline → optimized" if scores else "Δ in points"}</span></div>')


def grid(study):
    cells = STUDIES[study]
    head = '<th scope="col" class="corner">Task</th>'
    for t in TOPOS:
        sub = f'<small>{FW[t]}</small>' if study == "framework" else ""
        head += f'<th scope="col">{TLABEL[t]}{sub}</th>'
    head += '<th scope="col" class="avg">Average</th>'
    rows = []
    for dom in DOMAINS:
        rows.append(f'<tr class="dgroup"><th scope="rowgroup" colspan="{len(TOPOS) + 2}">{dom}</th></tr>')
        for t in [t for t in TASKS if t["domain"] == dom]:
            tid = t["id"]
            tds = "".join(cell_html(cells[tid][tp], tip(t["label"], TLABEL[tp] + (f" ({FW[tp]})" if study == "framework" else ""), t["metric"]))
                          for tp in TOPOS)
            tds += avg_cell([cells[tid][tp] for tp in TOPOS], tip(t["label"], "average over topologies"))
            rows.append(f'<tr><th scope="row">{t["label"]}<small>{t["metric"]}</small></th>{tds}</tr>')
    foot = "".join(avg_cell([cells[t["id"]][tp] for t in TASKS], tip(TLABEL[tp], "average over tasks"), cls="foot")
                   for tp in TOPOS)
    allv = [cells[t["id"]][tp] for t in TASKS for tp in TOPOS]
    foot += avg_cell(allv, tip("All cells"), cls="avg all")
    caption = ("Topology study: GEPA on the five topologies." if study == "topology" else
               "Framework study: GEPA on popular multi-agent frameworks, one framework per topology.")
    return (f'<div class="dgrid-wrap" data-study="{study}">\n<table class="dgrid">'
            f'<caption>{caption}</caption>\n<thead><tr>{head}</tr></thead>\n<tbody>\n' + "\n".join(rows) +
            f'\n</tbody>\n<tfoot><tr><th scope="row">Average</th>{foot}</tr></tfoot>\n</table>\n{legend()}\n</div>\n')


def task_strip(tid):
    t = TASK[tid]
    head = '<th scope="col" class="corner">Study</th>' + "".join(f'<th scope="col">{TLABEL[tp]}</th>' for tp in TOPOS)
    rows = []
    for study, label, sub in [("topology", "Topology study", ""), ("framework", "Framework study", "one framework per topology")]:
        cells = STUDIES[study][tid]
        tds = ""
        for tp in TOPOS:
            name = TLABEL[tp] + (f" ({FW[tp]})" if study == "framework" else "")
            extra = f'<em>{FW[tp]}</em>' if study == "framework" else ""
            c = cells[tp]
            d = delta(c)
            tds += (f'<td class="dc d-{step(d)}" data-tip="{tip(t["label"], name, t["metric"])}">'
                    f'<b>{fmt_d(d)}</b><span>{sc(c[0])} → {sc(c[1])}</span>{extra}</td>')
        rows.append(f'<tr><th scope="row">{label}{f"<small>{sub}</small>" if sub else ""}</th>{tds}</tr>')
    return (f'<div class="dgrid-wrap compact">\n<table class="dgrid">'
            f'<caption>{t["label"]}: GEPA gain by topology ({t["metric"]})</caption>\n<thead><tr>{head}</tr></thead>\n'
            f'<tbody>\n' + "\n".join(rows) + f'\n</tbody>\n</table>\n{legend()}\n</div>\n')


def dbars(rows, lo, hi, caption, cls="", head="Task"):
    """Diverging bar table. rows: (label, sublabel, delta, base, opt, group)."""
    span = hi - lo
    zero = (0 - lo) / span * 100
    has_scores = any(r[3] is not None for r in rows)
    step_ = 10 if span > 20 else 2
    ticks = [v for v in range(-100, 101, step_) if lo <= v <= hi and v != 0]
    tick_html = "".join(f'<i class="tick" style="left:{(v - lo) / span * 100:.2f}%"></i>' for v in ticks)
    out = [f'<div class="dbars-wrap {cls}">\n<table class="dbars">'
           f'<caption>{caption}</caption>\n'
           f'<thead><tr><th scope="col">{head}</th><th scope="col" class="bar-col">Gain (Δ, points)</th>'
           '<th scope="col" class="num">Δ</th>' + ('<th scope="col" class="num scores">Baseline → optimized</th>' if has_scores else '') +
           '</tr></thead>\n<tbody>']
    group = None
    for label, sub, d, b, o, g in rows:
        if g and g != group:
            out.append(f'<tr class="dgroup"><th scope="rowgroup" colspan="{4 if has_scores else 3}">{g}</th></tr>')
            group = g
        if d > 0:
            bar = f'<i class="bar pos" style="left:{zero:.2f}%;width:{d / span * 100:.2f}%"></i>'
        elif d < 0:
            w = -d / span * 100
            bar = f'<i class="bar neg" style="left:{zero - w:.2f}%;width:{w:.2f}%"></i>'
        else:
            bar = f'<i class="bar zero" style="left:{zero:.2f}%"></i>'
        sublabel = f"<small>{sub}</small>" if sub else ""
        scores = f'<td class="num scores">{b:.1f} → {o:.1f}</td>' if has_scores else ""
        out.append(f'<tr data-tip="{tip(label, sub, fmt_d(d) + " points")}"><th scope="row">{label}{sublabel}</th>'
                   f'<td class="bar-col"><span class="track">{tick_html}<i class="axis" style="left:{zero:.2f}%"></i>{bar}</span></td>'
                   f'<td class="num d-txt-{"g" if d > 0 else "l" if d < 0 else "z"}">{fmt_d(d)}</td>{scores}</tr>')
    out.append("</tbody>\n</table>\n</div>\n")
    return "\n".join(out)


def topology_bars(tp, study):
    cells = STUDIES[study]
    rows = []
    for t in TASKS:
        c = cells[t["id"]][tp]
        rows.append((t["label"], t["metric"], delta(c), c[0], c[1], t["domain"]))
    name = TLABEL[tp] + (f" ({FW[tp]})" if study == "framework" else "")
    avg = r1(mean(r[2] for r in rows))
    cap = f"{name}: GEPA gain on each task, {'framework' if study == 'framework' else 'topology'} study. Average {fmt_d(avg)} points."
    return dbars(rows, -17, 25, cap)


def summary_bars():
    out = {}
    # domains (framework study, as in the paper)
    fwc = STUDIES["framework"]
    rows = []
    for dom in DOMAINS:
        vals = [delta(fwc[t["id"]][tp]) for t in TASKS if t["domain"] == dom for tp in TOPOS]
        rows.append((dom, f"{sum(1 for t in TASKS if t['domain'] == dom)} tasks", r1(mean(vals)), None, None, None))
    out["summary-domains"] = dbars(rows, -3, 5, "Average gain by task domain (framework study).", "mini", "Domain")
    tsc = STUDIES["topology"]
    rows = []
    for tp in TOPOS:
        vals = [tsc[t["id"]][tp] for t in TASKS]
        rows.append((TLABEL[tp], "", r1(mean(delta(v) for v in vals)), None, None, None))
    out["summary-topologies"] = dbars(rows, -3, 5, "Average gain by topology (topology study).", "mini", "Topology")
    P = DATA["protocols"]
    rows = []
    for j, f in enumerate(P["formats"]):
        vals = [P["delta"][ds][i][j] for ds in P["datasets"] for i in range(len(P["topologies"]))]
        rows.append((f["label"], "", r1(mean(vals)), None, None, None))
    out["summary-protocols"] = dbars(rows, -3, 5, "Average gain by communication protocol (HotpotQA and LiveCodeBench).", "mini", "Protocol")
    T = DATA["team_sizes"]
    rows = []
    for j, n in enumerate(T["sizes"]):
        vals = [T["delta"][ds][i][j] for ds in T["datasets"] for i in range(len(T["topologies"]))]
        rows.append((f"n = {n}", "", r1(mean(vals)), None, None, None))
    out["summary-team-sizes"] = dbars(rows, -3, 5, "Average gain by team size (HotpotQA and LiveCodeBench).", "mini", "Team size")
    return out


def factor_grid(kind):
    """Heatmap panels for the protocol and team-size studies: topology × setting, one panel per dataset."""
    S = DATA["protocols"] if kind == "protocols" else DATA["team_sizes"]
    cols = ([f["label"] for f in S["formats"]] if kind == "protocols" else [f"n = {n}" for n in S["sizes"]])
    panels = []
    for ds in S["datasets"]:
        head = '<th scope="col" class="corner">Topology</th>' + "".join(f'<th scope="col">{c}</th>' for c in cols)
        rows = []
        for i, tp in enumerate(S["topologies"]):
            tds = ""
            for j, c in enumerate(cols):
                d = float(S["delta"][ds][i][j])
                tds += (f'<td class="dc d-{step(d)}" data-tip="{tip(TASK[ds]["label"], TLABEL[tp], c)}">'
                        f'<b>{fmt_d(d)}</b></td>')
            rows.append(f'<tr><th scope="row">{TLABEL[tp]}</th>{tds}</tr>')
        means = [mean(float(S["delta"][ds][i][j]) for i in range(len(S["topologies"]))) for j in range(len(cols))]
        foot = "".join(f'<td class="dc avg d-{step(r1(m))}" data-tip="{tip(TASK[ds]["label"], "mean over topologies", c)}"><b>{fmt_d(r1(m))}</b></td>'
                       for m, c in zip(means, cols))
        panels.append(f'<div class="dgrid-panel">\n<table class="dgrid narrow">'
                      f'<caption>{TASK[ds]["label"]}</caption>\n<thead><tr>{head}</tr></thead>\n<tbody>\n' + "\n".join(rows) +
                      f'\n</tbody>\n<tfoot><tr><th scope="row">Mean</th>{foot}</tr></tfoot>\n</table>\n</div>')
    return '<div class="dgrid-wrap panels">\n<div class="dgrid-panels">\n' + "\n".join(panels) + f'\n</div>\n{legend(False)}\n</div>\n'


def headline():
    """Key figures for the home page and the results page."""
    tsc, fwc = STUDIES["topology"], STUDIES["framework"]
    cells = [(t, tp, s, (fwc if s == "framework" else tsc)[t["id"]][tp]) for s in ("topology", "framework") for t in TASKS for tp in TOPOS]
    best = max(cells, key=lambda x: delta(x[3]))
    worst = min(cells, key=lambda x: delta(x[3]))
    study = [c for c in cells if c[2] == "topology"]
    up = sum(1 for c in study if delta(c[3]) > 0)
    down = sum(1 for c in study if delta(c[3]) < 0)
    flat = len(study) - up - down
    def name(c):
        t, tp, s, _ = c
        return f'{TLABEL[tp]}{" (" + FW[tp] + ")" if s == "framework" and tp in ("sequential", "centralized", "decentralized") else ""} · {t["label"]}'
    figs = [
        (fmt_d(delta(best[3])), "largest gain", name(best)),
        (fmt_d(delta(worst[3])), "largest drop", name(worst)),
        (f"{up} of {len(study)}", "cells improved", f"topology study · {down} regressed, {flat} unchanged"),
        (fmt_d(r1(mean(delta(tsc[t['id']]['single']) for t in TASKS))), "single-agent average", "every multi-agent topology gains less"),
    ]
    items = "".join(f'<div class="hfig"><b>{v}</b><span>{l}</span><small>{escape(s)}</small></div>' for v, l, s in figs)
    return f'<div class="hfigs">{items}</div>\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"grid-topology-study": grid("topology"), "grid-framework-study": grid("framework"),
             "protocols": factor_grid("protocols"), "team-sizes": factor_grid("team_sizes"), "headline": headline()}
    for t in TASKS:
        files[f"task-{t['id']}"] = task_strip(t["id"])
    for tp in TOPOS:
        files[f"topology-{tp}"] = topology_bars(tp, "topology")
        if tp in ("sequential", "centralized", "decentralized"):
            files[f"topology-{tp}-framework"] = topology_bars(tp, "framework")
    files.update(summary_bars())
    for name, html in files.items():
        assert "\n\n" not in html, name  # a blank line would end the raw HTML block
        (OUT / f"{name}.html").write_text(html)
    print(f"wrote {len(files)} components to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
