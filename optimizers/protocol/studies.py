"""Extension studies: experiment grids beyond the paper's tables, run only when named.

Four studies, each a set of :class:`~optimizers.protocol.cells.GridCell` (LangGraph,
Qwen task model, optimizer seeds 0-2):

* ``new_topologies`` -- the four primary methods on all nine tasks with the two
  extension topologies, Fan-out/Fan-in and Iterative Refinement (four agents).
* ``prompt_set2`` -- the four primary methods on the application tasks and the four
  multi-agent topologies, started from the second seed-prompt set P2.
* ``protocol_extension`` -- GEPA, MIPRO and MASPO on MATH, SWE-bench and API-Bank,
  the four multi-agent topologies and the three communication formats (r = 4).
* ``team_size_extension`` -- the same methods, tasks and topologies with freeform
  messages at r = 4, 8 and 10.

A study cell that is also a paper-grid cell (the freeform r = 4 cells of the last
two studies) is that very cell, with its source tables, so its job and results are
shared. ``run --study <name>`` admits the other cells; their ``job.json`` and
``result.json`` record the study, and ``aggregate`` reports them only for
``--study <name>``. List a study's jobs with::

    python -m optimizers.protocol.studies new_topologies --out-root runs/new_topologies
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Mapping, Sequence
from functools import cache
from itertools import product
from typing import Any

from .cells import (
    APPLICATION_TASKS,
    MULTI_TOPOLOGIES,
    PRIMARY_METHODS,
    QWEN_MODEL,
    TASKS,
    GridCell,
    build_grid,
    find_cell,
    registry_key,
    runner_condition_in_grid,
)
from .config import COMMUNICATIONS, DEFAULT_PROMPT_SET, EXTENSION_TOPOLOGIES, OPTIMIZER_SEEDS
from .schema import CellSpec

# Config
STUDIES = ("new_topologies", "prompt_set2", "protocol_extension", "team_size_extension")
EXTENSION_TASKS = ("math", "swe", "apibank")
EXTENSION_METHODS = ("gepa", "mipro", "maspo")
EXTENSION_TEAM_SIZES = (4, 8, 10)
CONDITION_FIELDS = ("task", "topology", "framework", "communication", "team_size", "task_model")
EXPECTED_STUDY_COUNTS = {
    "new_topologies": {"configurations": 72, "jobs": 216, "shared_with_paper_grid": 0},
    "prompt_set2": {"configurations": 48, "jobs": 144, "shared_with_paper_grid": 0},
    "protocol_extension": {"configurations": 108, "jobs": 324, "shared_with_paper_grid": 36},
    "team_size_extension": {"configurations": 108, "jobs": 324, "shared_with_paper_grid": 36},
}


def _conditions(name: str) -> list[tuple[Any, ...]]:
    """``(method, task, topology, communication, team size, prompt set)`` of every cell of ``name``."""
    if name == "new_topologies":
        return [
            (method, task, topology, "freeform", 4, DEFAULT_PROMPT_SET)
            for method, task, topology in product(PRIMARY_METHODS, TASKS, EXTENSION_TOPOLOGIES)
        ]
    if name == "prompt_set2":
        return [
            (method, task, topology, "freeform", 4, "P2")
            for method, task, topology in product(PRIMARY_METHODS, APPLICATION_TASKS, MULTI_TOPOLOGIES)
        ]
    if name == "protocol_extension":
        return [
            (method, task, topology, communication, 4, DEFAULT_PROMPT_SET)
            for method, task, topology, communication in product(
                EXTENSION_METHODS, EXTENSION_TASKS, MULTI_TOPOLOGIES, COMMUNICATIONS
            )
        ]
    if name == "team_size_extension":
        return [
            (method, task, topology, "freeform", size, DEFAULT_PROMPT_SET)
            for method, task, topology, size in product(
                EXTENSION_METHODS, EXTENSION_TASKS, MULTI_TOPOLOGIES, EXTENSION_TEAM_SIZES
            )
        ]
    raise ValueError(f"unknown study {name!r}; choices={STUDIES}")


@cache
def build_study(name: str) -> tuple[GridCell, ...]:
    """The cells of study ``name``; a cell of the paper grid is returned as that grid cell."""
    paper = build_grid(include_llama=True)
    cells = []
    for method, task, topology, communication, size, prompt_set in _conditions(name):
        args = (method, task, topology, "langgraph", communication, size, QWEN_MODEL, prompt_set)
        cells.append(find_cell(*args, cells=paper) or GridCell(*args[:7], prompt_set=prompt_set))
    return tuple(sorted(cells))


def find_study_cell(study: str, method: str, **condition: Any) -> GridCell | None:
    """The cell of ``study`` for ``method`` and one runtime condition (``find_cell`` keywords), or None."""
    return find_cell(method, **condition, cells=build_study(study))


def runner_condition_in_study(study: str, **condition: Any) -> bool:
    """Whether any cell of ``study`` uses this runtime condition."""
    return runner_condition_in_grid(**condition, cells=build_study(study))


def in_study(cell: CellSpec | GridCell, study: str | None = None) -> bool:
    """Whether ``cell`` is a cell of ``study`` (default: of any study)."""
    condition = {name: getattr(cell, name) for name in CONDITION_FIELDS}
    return any(
        find_study_cell(name, cell.method, **condition, prompt_set=cell.prompt_set) is not None
        for name in (STUDIES if study is None else (study,))
    )


def result_in_study(result: Mapping[str, Any], study: str) -> bool:
    """Whether a job result belongs to ``study`` (its own extension cell or a shared paper-grid cell)."""
    grid = result["grid_cell"]
    condition = {name: grid[name] for name in CONDITION_FIELDS}
    prompt_set = grid.get("prompt_set", DEFAULT_PROMPT_SET)
    return find_study_cell(study, grid["method"], **condition, prompt_set=prompt_set) is not None


def study_assertions() -> dict[str, dict[str, int]]:
    """Check every study's size against :data:`EXPECTED_STUDY_COUNTS`."""
    paper = {cell.identity for cell in build_grid(include_llama=True)}
    values = {}
    for name in STUDIES:
        cells = build_study(name)
        values[name] = {
            "configurations": len(cells),
            "jobs": len(cells) * len(OPTIMIZER_SEEDS),
            "shared_with_paper_grid": sum(cell.identity in paper for cell in cells),
        }
    if values != EXPECTED_STUDY_COUNTS:
        raise AssertionError(f"study grid mismatch: {values} != {EXPECTED_STUDY_COUNTS}")
    return values


# Job listing
def job_commands(study: str, *, out_root: str, model: str = "qwen") -> list[str]:
    """One ``optimizers.protocol.run`` command per job of ``study`` (each with its own ``--out``)."""
    commands = []
    for cell in build_study(study):
        key = registry_key(cell.topology, cell.framework, cell.communication, cell.team_size)
        prompt_set = [] if cell.prompt_set == DEFAULT_PROMPT_SET else [cell.prompt_set.lower()]
        for seed in OPTIMIZER_SEEDS:
            out = "/".join((out_root.rstrip("/"), cell.method, cell.task, key, *prompt_set, model, str(seed)))
            argv = ["--method", cell.method, "--dataset", cell.task, "--topology", key, "--model", model]
            argv += ["--seed", str(seed), "--study", study, "--out", out]
            if prompt_set:
                argv += ["--prompt-set", cell.prompt_set]
            commands.append(shlex.join(["python", "-m", "optimizers.protocol.run", *argv]))
    return commands


def main(argv: Sequence[str] | None = None) -> int:
    """Print the run commands of one study (or, with ``--counts``, every study's size)."""
    parser = argparse.ArgumentParser(description="List the jobs of an extension study (no model calls).")
    parser.add_argument("study", nargs="?", choices=STUDIES)
    parser.add_argument("--out-root", default="runs", help="root of the jobs' --out directories")
    parser.add_argument("--counts", action="store_true", help="print every study's configuration and job counts")
    args = parser.parse_args(argv)
    if args.counts or args.study is None:
        for name, counts in study_assertions().items():
            print(f"{name}\t" + "\t".join(f"{key}={value}" for key, value in counts.items()))
        return 0
    print("\n".join(job_commands(args.study, out_root=args.out_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
