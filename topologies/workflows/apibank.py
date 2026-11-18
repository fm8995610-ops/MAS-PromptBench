"""API-Bank under an extension topology (LangGraph): Fan-out/Fan-in or Iterative Refinement.

The topology's graph is :mod:`core.workflows`; every agent turn is one model call
with the role's system prompt on the dialogue and API descriptions, followed by
the turn's instruction (agent i of the team samples with seed i), and is scored
like any API-Bank agent. Reports reach other agents as in the other runners
(freeform). The prediction is the API call of the final turn: the coordinator's
synthesis, or the generator's latest candidate. Data, prompts, scoring and
records live in :mod:`core.tasks.apibank`. Executed with ``TOPOLOGY`` preset by
``topologies/<topology>/apibank/langgraph_apibank.py`` (:mod:`core.variant`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from core import agent_runs, settings, teams, workflows
from core.tasks import apibank as task
from core.tasks.apibank import (  # noqa: F401  (runner API)
    APIBANK_LEVEL,
    BENCHMARK_NAME,
    dataset_summary,
    extract_api_call,
    format_prompt,
    load_instances,
    normalize_level,
    parse_api_call,
    score_prediction,
)
from core.tasks.apibank import format_chat_history as _format_chat_history  # noqa: F401  (runner API)
from core.telemetry import sum_telemetry

DATASET_NAME = task.DATASET
TOPOLOGY = globals()["TOPOLOGY"]  # preset by the topologies/<topology>/apibank entries (core.variant)
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


def _load_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    return task.system_prompt(topology, role, style, prompt_suffix)


def solve(
    instance: dict,
    *,
    style: str,
    topology: str,
    role: str,
    seed: int | None = None,
    prompt_suffix: str = "",
    extra_context: str = "",
) -> dict:
    """One model call on ``instance``: ``{"messages", "raw", "solve_s", "telemetry"}``."""
    client = task.client(VLLM_BASE_URL)
    user = task.user_message(instance, extra_context)
    system = _load_prompt(topology, role, style, prompt_suffix)
    return task.complete(client, system, user, model=MODEL_ID, seed=seed)


def _render(turns: Sequence[workflows.Turn]) -> str:
    """Reports as the other runners hand them off (freeform), under their agent IDs."""
    reports = [{"role": turn.request.agent, "raw": workflows.report(turn)} for turn in turns]
    return agent_runs.reports_context(reports, dataset=task.DATASET)


def _turn_output(request: workflows.Request, agent: dict) -> dict:
    """A turn's entry of ``turn_outputs``: its agent ID, phase and round, and its compact scored agent."""
    return {"agent": request.agent, "phase": request.phase, "round": request.round, **task.compact_agent(agent)}


def solve_topology(
    instance: dict,
    *,
    style: str,
    topology: str,
    role: str,
    prompt_suffix: str = "",
    roles: list[str] | tuple[str, ...] | None = None,
    worker_roles: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Run :data:`TEAM` on one row (``role``, ``roles`` and ``worker_roles`` are the adapters' common arguments).

    Returns the prediction fields and ``raw`` of the final turn (empty without
    one), ``turn_outputs`` (each turn's scored agent), ``messages``
    (:func:`core.workflows.message_log`), the summed ``telemetry`` and the run
    fields of :func:`core.workflows.run_fields`.
    """
    agent_runs.check_topology(topology, TOPOLOGY)
    start = time.time()
    agents: dict[workflows.Request, dict] = {}  # the scored agent of each turn

    async def call(request: workflows.Request) -> workflows.Turn:
        agent = await asyncio.to_thread(
            task.solve_agent,
            solve,
            instance,
            style=style,
            topology=topology,
            role=request.role,
            seed=AGENTS.index(request.agent),
            prompt_suffix=prompt_suffix,
            extra_context=request.instruction,
        )
        agents[request] = agent
        artifact = agent.get("predicted_answer") if request.phase in workflows.ARTIFACT_PHASES else None
        return workflows.Turn(request, agent.get("raw") or "", artifact=artifact, messages=agent["messages"])

    run = asyncio.run(workflows.run_team(TEAM, call, render=_render))
    records = [agents[turn.request] for turn in run.turns]
    final = agents[run.final.request] if run.final else {}
    return {
        "topology": topology,
        "n_agents": TEAM.size,
        "turn_outputs": [_turn_output(turn.request, agents[turn.request]) for turn in run.turns],
        **task.answer_fields(final),
        "raw": final.get("raw") or "",
        "tool_calls": 0,
        "turns": len(records),
        "solve_s": round(time.time() - start, 1),
        "messages": workflows.message_log(run),
        "telemetry": sum_telemetry(records),
        **workflows.run_fields(run),
    }


def run_one(instance: dict, out_dir: Path, *, style: str, topology: str, role: str) -> dict:
    """The record of one row (:func:`core.tasks.apibank.run_instance`) plus the output's :data:`RECORD_KEYS`."""
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
    level: str | int | None = None,
) -> dict:
    """Solve and record the selected rows; ``out_dir`` defaults to ``results/apibank/<style>``."""
    instances = load_instances(limit=limit, offset=offset, only=only, level=level)
    run = partial(run_one, style=style, topology=topology, role=role)
    return task.run_rows(instances, run, style=style, model_id=MODEL_ID, out_dir=out_dir, level=level, verbose=verbose)


def main(argv: list[str] | None = None) -> int:
    run = partial(run_one, style=STYLE, topology=TOPOLOGY, role=ROLE)
    return task.main(argv, description=f"{BENCHMARK_NAME} runner", run_one=run, style=STYLE, model_id=MODEL_ID)


if __name__ == "__main__":
    raise SystemExit(main())
