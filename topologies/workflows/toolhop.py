"""ToolHop under an extension topology (LangGraph): Fan-out/Fan-in or Iterative Refinement.

The topology's graph is :mod:`core.workflows`; every agent turn is a tool loop
with the row's tools and the role's system prompt on the question, followed by
the turn's instruction (agent i of the team samples with seed i), and is scored
like any ToolHop agent. A turn that carries a candidate answer ends with an
``<answer>`` tag, as a final role's does; the others (plans, reports, reviews
and verdicts) keep their own reply. Reports reach other agents as in the other
runners (freeform). The prediction is the answer of the final turn: the
coordinator's synthesis, or the generator's latest candidate. Data, tools, the
tool loop, scoring and records live in :mod:`core.tasks.toolhop`. Executed with
``TOPOLOGY`` preset by ``topologies/<topology>/toolhop/langgraph_toolhop.py``
(:mod:`core.variant`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from core import agent_runs, settings, teams, workflows
from core.tasks import toolhop as task
from core.tasks.toolhop import (  # noqa: F401  (runner API)
    DEFAULT_MAX_TURNS,
    HF_DATASET,
    dataset_summary,
    extract_answer,
    load_instances,
    score_answer,
)
from core.tasks.toolhop import last_assistant_content as _last_assistant_content  # noqa: F401  (runner API)
from core.tasks.toolhop import previous_tool_content as _previous_tool_content  # noqa: F401  (runner API)
from core.telemetry import sum_telemetry

TOPOLOGY = globals()["TOPOLOGY"]  # preset by the topologies/<topology>/toolhop entries (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET)
STYLE = f"{TOPOLOGY}_langgraph"
ROLE = TEAM.roles[0]  # the final role: the coordinator or the generator
# Agent IDs in team order: agent i samples with seed i.
if TOPOLOGY == "fan_out_fan_in":
    AGENTS = ("coordinator", *(f"worker_{i}" for i in range(1, TEAM.n_workers + 1)))
else:
    AGENTS = TEAM.roles
# Output keys the record keeps besides the task's own: the turns and how the run ended.
RECORD_KEYS = ("turn_outputs", "termination", "rounds", "subtasks", "verdicts", "n_turns")

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()


def _system_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    return task.system_prompt(topology, role, style, prompt_suffix)


def solve(
    sample: dict,
    *,
    style: str,
    topology: str,
    role: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    seed: int | None = None,
    prompt_suffix: str = "",
    extra_context: str = "",
    candidate: bool = False,
) -> dict:
    """Run one tool loop on ``sample``: ``{"messages", "solve_s", "telemetry"}``.

    A ``candidate`` turn (one that carries a candidate final answer) ends with an
    ``<answer>`` tag, forced if its reply has none; any other turn keeps its reply
    unless the tool-call budget runs out.
    """
    functions = task.function_map(sample)
    client = task.client(VLLM_BASE_URL)
    user = task.user_message(sample, extra_context)
    system = _system_prompt(topology, role, style, prompt_suffix)
    # The tool loop forces the answer of the roles in task.FINAL_ANSWER_ROLES: "solver" is one, "worker" is not.
    loop_role = "solver" if candidate else "worker"
    return task.tool_loop(
        client, sample, functions, system, user, model=MODEL_ID, role=loop_role, seed=seed, max_turns=max_turns
    )


def _render(turns: Sequence[workflows.Turn]) -> str:
    """Reports as the other runners hand them off (freeform), under their agent IDs."""
    reports = [{"role": turn.request.agent, "final_content": workflows.report(turn)} for turn in turns]
    return agent_runs.reports_context(reports, dataset=task.DATASET)


def _turn_output(request: workflows.Request, agent: dict) -> dict:
    """A turn's entry of ``turn_outputs``: its agent ID, phase and round, and its compact scored agent."""
    return {"agent": request.agent, "phase": request.phase, "round": request.round, **task.compact_agent(agent)}


def solve_topology(
    sample: dict,
    *,
    style: str,
    topology: str,
    role: str,
    prompt_suffix: str = "",
    roles: list[str] | tuple[str, ...] | None = None,
    worker_roles: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Run :data:`TEAM` on one row (``role``, ``roles`` and ``worker_roles`` are the adapters' common arguments).

    Returns the prediction fields, ``final_content`` and ``previous_tool_content``
    of the final turn (empty without one), ``turn_outputs`` (each turn's scored
    agent), ``messages`` (:func:`core.workflows.message_log`), the summed
    ``telemetry`` and the run fields of :func:`core.workflows.run_fields`.
    """
    agent_runs.check_topology(topology, TOPOLOGY)
    start = time.time()
    agents: dict[workflows.Request, dict] = {}  # the scored agent of each turn

    async def call(request: workflows.Request) -> workflows.Turn:
        agent = await asyncio.to_thread(
            task.solve_agent,
            solve,
            sample,
            style=style,
            topology=topology,
            role=request.role,
            seed=AGENTS.index(request.agent),
            prompt_suffix=prompt_suffix,
            extra_context=request.instruction,
            candidate=request.phase in workflows.ARTIFACT_PHASES,
        )
        agents[request] = agent
        artifact = agent.get("predicted_answer") if request.phase in workflows.ARTIFACT_PHASES else None
        return workflows.Turn(request, agent.get("final_content") or "", artifact=artifact, messages=agent["messages"])

    run = asyncio.run(workflows.run_team(TEAM, call, render=_render))
    records = [agents[turn.request] for turn in run.turns]
    final = agents[run.final.request] if run.final else {}
    return {
        "topology": topology,
        "n_agents": TEAM.size,
        "turn_outputs": [_turn_output(turn.request, agents[turn.request]) for turn in run.turns],
        **task.answer_fields(final),
        "final_content": final.get("final_content") or "",
        "previous_tool_content": final.get("previous_tool_content") or "",
        **task.usage(records),
        "solve_s": round(time.time() - start, 1),
        "messages": workflows.message_log(run),
        "telemetry": sum_telemetry(records),
        **workflows.run_fields(run),
    }


def run_one(instance: dict, out_dir: Path, *, style: str, topology: str, role: str) -> dict:
    """The record of one row (:func:`core.tasks.toolhop.run_instance`) plus the output's :data:`RECORD_KEYS`."""
    out: dict = {}

    def solve_row() -> dict:
        out.update(solve_topology(instance, style=style, topology=topology, role=role))
        return out

    record = task.run_instance(instance, out_dir, style=style, solve=solve_row)
    return {**record, **{key: out[key] for key in RECORD_KEYS if key in out}}


def run_batch(
    *,
    style: str,
    topology: str,
    role: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str | int] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Solve and record the selected rows; ``out_dir`` defaults to ``results/toolhop/<style>``."""
    instances = load_instances(limit=limit, offset=offset, only=only)
    run = partial(run_one, style=style, topology=topology, role=role)
    return task.run_rows(instances, run, style=style, model_id=MODEL_ID, out_dir=out_dir, verbose=verbose)


def main(argv: list[str] | None = None) -> int:
    run = partial(run_one, style=STYLE, topology=TOPOLOGY, role=ROLE)
    return task.main(argv, description=f"ToolHop runner ({STYLE}).", run_one=run, style=STYLE, model_id=MODEL_ID)


if __name__ == "__main__":
    raise SystemExit(main())
