"""SWE-bench Verified under an extension topology (LangGraph): Fan-out/Fan-in or Iterative Refinement.

The topology's graph is :mod:`core.workflows`; every agent turn is a ReAct agent
with the role's prompt and tools (``configs/teams/swe.yaml``) on the issue brief,
followed by the turn's instruction. The run's checkout is the one bound by
``_set_repo_dir``. Fan-out/Fan-in: each worker works in its own clone of it, made
at the start of its turn, and reports that clone's ``git diff HEAD``; the
coordinator plans and synthesizes in the bound checkout, where it applies the
combined fix. Iterative Refinement: the generator edits the bound checkout, whose
diff after its turn is the candidate; the critics and the verifier have
read-only tools on it. The patch is the final turn's diff of the bound checkout
(none after an invalid decomposition). Executed with ``TOPOLOGY`` preset by
``topologies/<topology>/swe/langgraph_swe.py`` (:mod:`core.variant`).
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from core import prompts, settings, swe_sandbox, teams, workflows
from core.llm import chat_openai
from core.tasks import swe as task
from core.tasks.swe import (  # noqa: F401  (runner API)
    clone_and_checkout,
    is_resolved,
    load_instances,
    run_tests_singularity,
)
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_ai_thinking

TOPOLOGY = globals()["TOPOLOGY"]  # preset by the topologies/<topology>/swe entries (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET)
MODEL_NAME = f"mas-promptbench-{TOPOLOGY}"
DEFAULT_WORKDIR_ROOT, DEFAULT_OUT_DIR = task.default_dirs(f"{TOPOLOGY}_langgraph")

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
_RECURSION_LIMIT = TEAM.recursion_limit
# The only tools of a read-only turn (critics and verifier), so that it cannot change the candidate.
READ_ONLY_TOOLS = frozenset({"file_read", "list_dir", "search_repo"})
# Ends every brief: the final roles' contract asks for a diff block, but the patch is the checkout's state.
PATCH_NOTE = (
    "The patch is `git diff HEAD` of this checkout: only edits made in it with the tools count, "
    "and a diff written in a reply is not applied."
)

WORKDIR = task.Workdir()  # bound by each turn to its agent's checkout
TOOLS = {
    "file_read": tool(task.make_file_read(WORKDIR, task.FILE_READ_DOC_SHORT)),
    "str_replace": tool(task.make_str_replace(WORKDIR, task.STR_REPLACE_DOC_NARROW)),
    "list_dir": tool(task.make_list_dir(WORKDIR, task.LIST_DIR_DOC)),
    "search_repo": tool(task.make_search_repo(WORKDIR, task.SEARCH_REPO_DOC, terse=True)),
    "shell_exec": tool(task.make_shell_exec(WORKDIR, task.SHELL_EXEC_DOC)),
}

_ensure_sif = task.ensure_sif


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_llm() -> ChatOpenAI:
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _set_repo_dir(path: Path | str) -> None:
    """Bind the run's checkout (in this context)."""
    WORKDIR.bind(Path(path).resolve())


def format_task_brief(problem_statement: str, instance_id: str | None = None, hints_text: str | None = None) -> str:
    """An agent's issue brief, at the bound checkout."""
    return task.issue_brief(
        problem_statement,
        instance_id,
        hints_text,
        checkout=task.checked_out_at(WORKDIR.get()),
        note=f"{task.NO_TESTS_NOTE}\n\n{PATCH_NOTE}",
    )


def compute_patch() -> str:
    """``git diff HEAD`` of the bound checkout."""
    return task.compute_patch(WORKDIR.get())


def predictions_entry(instance_id: str, patch: str, model_name: str = MODEL_NAME) -> dict:
    return task.predictions_entry(instance_id, patch, model_name)


@contextmanager
def _worker_root(main: Path, instance_id: str | None):
    """A directory beside ``main`` for the fan-out workers' clones, sandboxed with the instance image; removed after."""
    root = Path(tempfile.mkdtemp(prefix=f"{main.name}_workers_", dir=main.parent))
    if instance_id:  # shell_exec finds the image of a clone through its parent directory
        swe_sandbox.register_worktree(root, lambda: _ensure_sif(instance_id))
    try:
        yield root
    finally:
        swe_sandbox.unregister_worktree(root)
        shutil.rmtree(root, ignore_errors=True)


def _clone(main: Path, target: Path) -> Path:
    """``target``: an independent clone of ``main`` at its checked-out commit (committed files only)."""
    head = subprocess.run(["git", "-C", str(main), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
    if head.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed in {main}: {head.stderr.strip()}")
    error = task.local_clone(main, target, head.stdout.strip())
    if error:
        raise RuntimeError(f"{target.name}: {error}")
    return target


async def _call(instance: dict, main: Path, workers: Path | None, request: workflows.Request) -> workflows.Turn:
    """One turn: the role's ReAct agent in its checkout, on the issue brief and the turn's instruction.

    A fan-out worker works in its own clone of ``main`` (under ``workers``), every other agent in ``main``.
    The checkout is bound in this turn's context, which the agent's tool calls inherit (LangGraph runs tasks,
    and LangChain sync tools, in copies of it), so parallel turns never act on each other's checkout. The
    artifact of a worker or artifact-phase turn is ``git diff HEAD`` of its checkout after the turn (a
    worker's clone is discarded, so its diff reaches the coordinator only as its report's artifact).
    """
    names = TEAM.tools(request.role)
    if request.read_only and not READ_ONLY_TOOLS.issuperset(names):
        raise ValueError(f"read-only role {request.role!r} has editing tools: {names}")
    worker = request.role == "worker"
    checkout = await asyncio.to_thread(_clone, main, workers / request.agent) if worker else main
    token = WORKDIR.bind(checkout)
    try:
        agent = create_react_agent(
            model=_build_llm(), tools=[TOOLS[name] for name in names], prompt=_load_prompt(request.role)
        )
        brief = format_task_brief(
            instance["problem_statement"], instance.get("instance_id"), instance.get("hints_text")
        )
        user = f"{brief}\n\n{request.instruction}"
        result = await agent.ainvoke({"messages": [("user", user)]}, config={"recursion_limit": _RECURSION_LIMIT})
    finally:
        WORKDIR.reset(token)
    strip_ai_thinking(result["messages"])
    text = result["messages"][-1].content or ""
    artifact = None
    if worker or request.phase in workflows.ARTIFACT_PHASES:
        artifact = await asyncio.to_thread(task.compute_patch, checkout)
    return workflows.Turn(request, text, artifact=artifact, messages=result["messages"])


def solve(instance: dict, eval_mode: str = "singularity") -> dict:
    """Run the team on the instance checked out at the bound checkout.

    Returns ``{"patch", "resolved", "report", "raw", "messages", "turn_patches",
    "telemetry", "termination", "rounds", "subtasks", "verdicts", "n_turns"}``:
    the final turn's patch, its evaluation (``resolved`` None and no report with
    ``eval_mode='none'``; False without a patch), the final turn's text, every
    turn as ``{source, agent, phase, round, content}`` with its patch alongside
    (None for a turn without an artifact), token/call counts and how the run ended.
    """
    main = WORKDIR.get()
    workers = _worker_root(main, instance.get("instance_id")) if TOPOLOGY == "fan_out_fan_in" else nullcontext()
    with workers as root:
        run = asyncio.run(workflows.run_team(TEAM, partial(_call, instance, main, root)))
    final = run.final
    out = {
        "patch": (final.artifact or "") if final else "",
        "resolved": None if eval_mode == "none" else False,
        "report": None,
        "raw": final.text if final else "",
        "messages": workflows.message_log(run),
        "turn_patches": [turn.artifact for turn in run.turns],
        "telemetry": normalize(langchain_telemetry(m for turn in run.turns for m in turn.messages)),
        **workflows.run_fields(run),
    }
    if eval_mode != "none" and out["patch"]:
        f2p, p2p = task.instance_tests(instance)
        out["report"] = run_tests_singularity(instance, out["patch"], f2p, p2p)
        out["resolved"] = is_resolved(out["report"])
    return out


def _trace(out: dict) -> str:
    """How the run ended, then every turn's text and, for an artifact turn, its patch."""
    pairs = (
        (
            f"{m['agent'].upper()} / {m['phase']} / round {m['round']}",
            m["content"] if patch is None else f"{m['content']}\n\n--- PATCH ---\n{patch}",
        )
        for m, patch in zip(out["messages"], out["turn_patches"])
    )
    return f"termination: {out['termination']}  rounds: {out['rounds']}\n\n" + task.sections(pairs)


def run_one(instance: dict, workdir_root: Path, out_dir: Path, eval_mode: str = "singularity") -> dict:
    """Clone, solve and score one instance; writes its patch, prediction and turn trace under ``out_dir``."""
    summary, out = task.solve_in_checkout(
        instance, workdir_root, _set_repo_dir, lambda: solve(instance, eval_mode=eval_mode)
    )
    if out is None:
        return summary
    summary.update(out.get("telemetry") or {})
    patch = out["patch"] or ""
    summary.update(patch_chars=len(patch), termination=out["termination"], rounds=out["rounds"], n_turns=out["n_turns"])
    iid = instance["instance_id"]
    task.write_artifacts(out_dir, iid, patch, predictions_entry(iid, patch), _trace(out))
    return {**summary, **task.eval_fields(eval_mode, out.get("report"))}


def _run_instances(
    instances: list[dict],
    out_dir: Path | None = None,
    workdir_root: Path | None = None,
    eval_mode: str = "singularity",
    keep_workdirs: bool = False,
    out_path: Path | None = None,
) -> None:
    """Solve and score loaded instances (the command line's batch)."""
    workdir_root = workdir_root or DEFAULT_WORKDIR_ROOT
    out_dir = out_dir or DEFAULT_OUT_DIR
    task.run_batch(
        instances,
        lambda inst: run_one(inst, workdir_root, out_dir, eval_mode=eval_mode),
        out_dir=out_dir,
        eval_mode=eval_mode,
        workdirs=None if keep_workdirs else lambda inst: [workdir_root / inst["instance_id"]],
        predictions=out_path,
    )


def run_batch(
    subset: str = "test",
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    workdir_root: Path | None = None,
    out_dir: Path | None = None,
    eval_mode: str = "singularity",
    keep_workdirs: bool = False,
) -> None:
    """Solve and score a Verified slice (``eval_mode``: ``singularity`` or ``none``)."""
    instances = load_instances(subset, limit, offset, only)
    task.log_loaded(instances)
    _run_instances(instances, out_dir, workdir_root, eval_mode, keep_workdirs)


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description=f"{TOPOLOGY} SWE-bench Verified agent (LangGraph).",
        run_batch=_run_instances,
        default_out_dir=DEFAULT_OUT_DIR,
        eval_modes=("singularity", "none"),
        skip_eval=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
