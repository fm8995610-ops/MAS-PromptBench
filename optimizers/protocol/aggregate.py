"""Aggregate job results over optimizer seeds 0/1/2 into paired summaries.

    python -m optimizers.protocol.aggregate runs/ [--out summary.json]

Per cell (method x task x topology x framework x communication x team size x
model): per-seed baseline / deployed / delta, mean and std of the delta, the
seed-stratified paired bootstrap CI, per-seed exact McNemar p-values (binary
scores) with Holm correction inside each family (default: dataset x model),
and the number of baseline fallbacks. A cell started from another seed-prompt
set (P2) is its own cell.

Extension-study jobs (``studies.py``) are reported only with ``--study <name>``,
which keeps that study's results (shared paper-grid cells included).
``--group-by`` adds one line per group of cells (e.g. ``task,communication``):
per seed the mean delta over the group's cells, then the mean and sample std of
those three seed means.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from core import logs

from .artifacts import atomic_write_json, read_json
from .config import DEFAULT_PROMPT_SET, OPTIMIZER_SEEDS, PROTOCOL_ID
from .reporting import PairedObservation, adjust_predeclared_families, summarize_paired_observations
from .schema import schema_name, verify_sealed
from .studies import STUDIES, result_in_study

logger = logging.getLogger(__name__)

GRID_FIELDS = ("method", "task", "topology", "framework", "communication", "team_size", "task_model")
GROUP_FIELDS = (*GRID_FIELDS, "prompt_set")


def find_results(roots: Iterable[Path]) -> list[Path]:
    """Every ``result.json`` under the given job directories or roots, sorted."""
    paths: list[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_file() and root.name == "result.json":
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.rglob("result.json")))
    return sorted(set(paths))


def _job_name(path: Path, roots: Sequence[Path]) -> str:
    """The job directory of ``path`` (a ``result.json``), relative to the root in ``roots`` that holds it.

    A job directory that is itself a root, or under none of them, is named as given.
    """
    job = Path(path).parent
    for root in map(Path, roots):
        if job != root and job.is_relative_to(root):
            return job.relative_to(root).as_posix()
    return job.as_posix()


def load_results(
    paths: Iterable[Path],
    *,
    include_nonconformant: bool = False,
    roots: Sequence[Path] = (),
    study: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Verified job results of this protocol, and a reason for every file skipped.

    A reason starts with the job directory, relative to the root in ``roots`` (the
    directories searched) that holds it. Without ``study`` the extension-study jobs
    are skipped; with it, every result outside that study is.
    """
    results, skipped = [], []
    for path in paths:
        name = _job_name(path, roots)
        try:
            value = verify_sealed(read_json(path))
        except Exception as exc:
            skipped.append(f"{name}: unreadable ({exc})")
            continue
        if value.get("schema") != schema_name("job-result") or value.get("protocol_id") != PROTOCOL_ID:
            skipped.append(f"{name}: not a {PROTOCOL_ID} job result")
            continue
        if not value.get("protocol_conformant") and not include_nonconformant:
            skipped.append(f"{name}: non-conformant job (budget, cell or reflection model outside the protocol)")
            continue
        if study is None and value.get("study"):
            skipped.append(f"{name}: job of the {value['study']} study (aggregate it with --study {value['study']})")
            continue
        if study is not None and not result_in_study(value, study):
            skipped.append(f"{name}: not a cell of the {study} study")
            continue
        results.append(value)
    return results, skipped


def cell_key(result: Mapping[str, Any]) -> tuple:
    """The grid identity a result is aggregated under (with the prompt set of a non-P1 cell)."""
    grid = result["grid_cell"]
    key = tuple(grid[name] for name in GRID_FIELDS)
    return (*key, grid["prompt_set"]) if "prompt_set" in grid else key


def summarize_cell(results: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int) -> dict[str, Any]:
    """Paired summary of one cell over its seeds (``incomplete`` with the problems when a seed is missing or invalid)."""
    by_seed: dict[int, Mapping[str, Any]] = {}
    duplicates = []
    for result in results:
        seed = int(result["optimizer_seed"])
        if seed in by_seed:
            duplicates.append(seed)
        by_seed[seed] = result
    base = {
        "cell": dict(zip(GROUP_FIELDS, cell_key(results[0]))),
        "registry_key": results[0].get("registry_key"),
        "source_tables": sorted({table for r in results for table in r.get("source_tables", [])}),
        "seeds_present": sorted(by_seed),
        "per_seed_job": {
            str(seed): {
                "selected_candidate": r["selection"]["selected_candidate"],
                "fallback_reason": r["selection"]["fallback_reason"],
                "baseline_validation": r["selection"]["baseline_validation_score"],
                "incumbent_validation": r["selection"]["incumbent_validation_score"],
                "charged_rollouts": (r["optimization"].get("budget") or {}).get("charged"),
                "stop_reason": r["optimization"].get("stop_reason"),
                "optimization_status": r["optimization"].get("status"),
                "valid_for_aggregation": r["test"]["valid_for_aggregation"],
            }
            for seed, r in sorted(by_seed.items())
        },
        "fallbacks": sum(1 for r in by_seed.values() if not r["selection"]["selected_candidate"]),
    }
    problems = []
    if duplicates:
        problems.append(f"duplicate results for seeds {sorted(set(duplicates))}")
    missing = [seed for seed in OPTIMIZER_SEEDS if seed not in by_seed]
    if missing:
        problems.append(f"missing seeds {missing}")
    invalid = [seed for seed, r in by_seed.items() if not r["test"]["valid_for_aggregation"]]
    if invalid:
        problems.append(f"infrastructure-invalid test data for seeds {sorted(invalid)}")
    if problems:
        return {**base, "status": "incomplete", "problems": problems}
    observations = [
        PairedObservation(seed, example_id, float(b), float(d))
        for seed, r in sorted(by_seed.items())
        for example_id, b, d in zip(
            r["test"]["example_ids"], r["test"]["baseline_scores"], r["test"]["deployed_scores"]
        )
    ]
    try:
        summary = summarize_paired_observations(observations, bootstrap_replicates=bootstrap_replicates)
    except ValueError as exc:
        return {**base, "status": "incomplete", "problems": [str(exc)]}
    return {**base, "status": "complete", **summary}


def family_name(cell: Mapping[str, Any], family: str) -> str:
    """Holm family label of a cell: ``task|model`` (default), the method, or one global family."""
    if family == "none":
        return "all"
    if family == "method":
        return f"{cell['method']}"
    return f"{cell['task']}|{cell['task_model']}"


def summarize_groups(cells: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    """One summary per group of cells sharing ``fields``: seed-level means, then their mean and sample std.

    For each optimizer seed the group's value is the mean over its cells of that
    seed's baseline, deployed score and delta; the group's delta is the mean of
    the three seed deltas and its std their sample std (not a mean of cell stds).
    A group with an incomplete cell is reported as incomplete.
    """
    unknown = sorted(set(fields) - set(GROUP_FIELDS))
    if unknown:
        raise ValueError(f"unknown group-by fields {unknown}; choices={GROUP_FIELDS}")
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        grid = {"prompt_set": DEFAULT_PROMPT_SET, **cell["cell"]}
        groups[tuple(grid[name] for name in fields)].append(cell)
    summaries = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        base = {"group": dict(zip(fields, key)), "n_cells": len(members)}
        incomplete = sum(1 for cell in members if cell["status"] != "complete")
        if incomplete:
            summaries.append({**base, "status": "incomplete", "incomplete_cells": incomplete})
            continue
        per_seed = []
        for seed in OPTIMIZER_SEEDS:
            rows = [row for cell in members for row in cell["per_seed"] if row["seed"] == seed]
            per_seed.append(
                {"seed": seed, **{k: mean(row[k] for row in rows) for k in ("baseline", "deployed", "delta_pp")}}
            )
        deltas = [row["delta_pp"] for row in per_seed]
        summaries.append(
            {
                **base,
                "status": "complete",
                "per_seed": per_seed,
                "macro_baseline": mean(row["baseline"] for row in per_seed),
                "macro_deployed": mean(row["deployed"] for row in per_seed),
                "mean_delta_pp": mean(deltas),
                "std_delta_pp": stdev(deltas),
            }
        )
    return summaries


def aggregate(
    results: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
    family: str = "task",
    group_by: Sequence[str] = (),
) -> dict[str, Any]:
    """Per-cell paired summaries with Holm-adjusted per-seed McNemar p-values (and ``group_by`` summaries)."""
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        groups[cell_key(result)].append(result)
    cells = [summarize_cell(items, bootstrap_replicates=bootstrap_replicates) for _, items in sorted(groups.items())]
    p_values: dict[str, float] = {}
    families: dict[str, list[str]] = defaultdict(list)
    for index, cell in enumerate(cells):
        for seed, value in (cell.get("per_seed_exact_mcnemar_p") or {}).items():
            label = f"{index}|seed={seed}"
            p_values[label] = value
            families[family_name(cell["cell"], family)].append(label)
    adjusted = adjust_predeclared_families(p_values, families) if p_values else {}
    for index, cell in enumerate(cells):
        holm = {}
        for name, values in adjusted.items():
            for label, value in values.items():
                if label.startswith(f"{index}|"):
                    holm[label.split("seed=", 1)[1]] = value
                    cell["holm_family"] = name
        if holm:
            cell["per_seed_mcnemar_p_holm"] = holm
    summary = {
        "schema": schema_name("aggregate"),
        "protocol_id": PROTOCOL_ID,
        "family": family,
        "bootstrap_replicates": bootstrap_replicates,
        "cells": cells,
    }
    if group_by:
        summary["group_by"] = list(group_by)
        summary["groups"] = summarize_groups(cells, group_by)
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def format_table(summary: Mapping[str, Any]) -> str:
    """Tab-separated summary table with one detail line per seed."""
    header = (
        "method",
        "task",
        "runtime",
        "model",
        "seeds",
        "base",
        "deployed",
        "delta_pp",
        "std_pp",
        "ci95_pp",
        "fallbacks",
        "min_p_holm",
    )
    lines = ["\t".join(header)]
    for cell in summary["cells"]:
        grid = cell["cell"]
        runtime = cell.get("registry_key") or grid["topology"]
        if "prompt_set" in grid:
            runtime += f" [{grid['prompt_set']}]"
        if cell["status"] != "complete":
            lines.append(
                "\t".join(
                    (
                        grid["method"],
                        grid["task"],
                        runtime,
                        grid["task_model"],
                        ",".join(map(str, cell["seeds_present"])),
                        "incomplete: " + "; ".join(cell["problems"]),
                    )
                )
            )
            continue
        ci = cell["confidence_interval_delta_pp"]
        holm = cell.get("per_seed_mcnemar_p_holm") or {}
        lines.append(
            "\t".join(
                (
                    grid["method"],
                    grid["task"],
                    runtime,
                    grid["task_model"],
                    "0,1,2",
                    _fmt(cell["macro_baseline"]),
                    _fmt(cell["macro_deployed"]),
                    _fmt(cell["mean_delta_pp"], 2),
                    _fmt(cell["std_delta_pp"], 2),
                    f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                    f"{cell['fallbacks']}/3",
                    _fmt(min(holm.values()) if holm else None),
                )
            )
        )
        for row in cell["per_seed"]:
            lines.append(
                f"\t  seed {row['seed']}: base={row['baseline']:.3f} deployed={row['deployed']:.3f} "
                f"delta={row['delta_pp']:+.2f}pp n={row['n']} "
                f"fallback={cell['per_seed_job'][str(row['seed'])]['fallback_reason']}"
            )
    for group in summary.get("groups", ()):
        label = " ".join(f"{name}={value}" for name, value in group["group"].items())
        if group["status"] != "complete":
            lines.append(f"group {label}\tcells={group['n_cells']}\tincomplete: {group['incomplete_cells']} cell(s)")
            continue
        lines.append(
            f"group {label}\tcells={group['n_cells']}\tbase={group['macro_baseline']:.3f} "
            f"deployed={group['macro_deployed']:.3f} delta={group['mean_delta_pp']:+.2f} +- {group['std_delta_pp']:.2f}pp"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate job results from the command line; exit code 1 when none is found."""
    parser = argparse.ArgumentParser(description="Paired seed-0/1/2 summary of protocol job results.")
    parser.add_argument("roots", nargs="+", type=Path, help="job directories or roots containing them")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON summary here")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="bootstrap replicates (>= 100)")
    parser.add_argument(
        "--family",
        choices=("task", "method", "none"),
        default="task",
        help="Holm family: dataset x model (default), method, or one family",
    )
    parser.add_argument("--include-nonconformant", action="store_true", help="include smoke-budget or off-grid jobs")
    parser.add_argument("--study", choices=STUDIES, help="report this extension study (studies.py) instead of the grid")
    parser.add_argument(
        "--group-by",
        default="",
        help=f"comma-separated fields of {GROUP_FIELDS}: add one seed-level summary per group of cells",
    )
    logs.add_argument(parser)
    args = parser.parse_args(argv)
    logs.configure(args.log_level)
    results, skipped = load_results(
        find_results(args.roots), include_nonconformant=args.include_nonconformant, roots=args.roots, study=args.study
    )
    for message in skipped:
        logger.warning("[aggregate] skipped %s", message)
    if not results:
        logger.error("[aggregate] no job results found")
        return 1
    group_by = tuple(name for name in args.group_by.split(",") if name)
    summary = aggregate(results, bootstrap_replicates=args.bootstrap, family=args.family, group_by=group_by)
    print(format_table(summary))
    if args.out is not None:
        atomic_write_json(args.out, summary, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
