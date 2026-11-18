"""APPS under an extension topology (LangGraph): Fan-out/Fan-in or Iterative Refinement.

The topology's graph is :mod:`core.workflows`; every agent turn is a python_exec
ReAct agent with the role's prompt and tools (``configs/teams/apps.yaml``) on the
problem (call-based mode with starter code, standard input without), followed by
the turn's instruction. The submission is the last fenced Python block of the
final turn: the coordinator's synthesis, or the generator's latest candidate.
Executed with ``TOPOLOGY`` preset by ``topologies/<topology>/apps/langgraph_apps.py``
(:mod:`core.variant`).
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core import cli, code_tasks, prompts, settings, teams, workflows
from core.batch import attempt
from core.code_tasks import exact_match_score, extract_code  # noqa: F401  (runner API)
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import apps as task
from core.tasks.apps import format_prompt, load_instances, run_tests  # noqa: F401  (runner API)
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_ai_thinking

TOPOLOGY = globals()["TOPOLOGY"]  # preset by the topologies/<topology>/apps entries (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET)
DEFAULT_PREDICTIONS = RESULTS_DIR / f"apps_{TOPOLOGY}_langgraph" / "predictions.jsonl"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
_RECURSION_LIMIT = TEAM.recursion_limit

python_exec = tool(code_tasks.make_python_exec(code_tasks.PYTHON_EXEC_DOC_SHORT))
TOOLS = {"python_exec": python_exec}


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_llm():
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


async def _call(problem: str, starter_code: str | None, request: workflows.Request) -> workflows.Turn:
    """One turn: the role's ReAct agent on the problem and the turn's instruction."""
    agent = create_react_agent(
        model=_build_llm(),
        tools=[TOOLS[name] for name in TEAM.tools(request.role)],
        prompt=_load_prompt(request.role),
    )
    user = f"{format_prompt(problem, starter_code)}\n\n{request.instruction}"
    result = await agent.ainvoke({"messages": [("user", user)]}, config={"recursion_limit": _RECURSION_LIMIT})
    strip_ai_thinking(result["messages"])
    text = result["messages"][-1].content or ""
    artifact = extract_code(text) if request.phase in workflows.ARTIFACT_PHASES else None
    return workflows.Turn(request, text, artifact=artifact, messages=result["messages"])


def solve(problem: str, starter_code: str | None = None) -> dict:
    """Run the team on one problem (call-based mode with ``starter_code``, standard input without).

    Returns ``{"code", "raw", "messages", "telemetry", "termination", "rounds",
    "subtasks", "verdicts", "n_turns"}``: the final turn's program (None without
    one) and text, every turn as ``{source, agent, phase, round, content}``,
    token/call counts and how the run ended.
    """
    run = asyncio.run(workflows.run_team(TEAM, partial(_call, problem, starter_code)))
    return {
        "code": run.final.artifact if run.final else None,
        "raw": run.final.text if run.final else "",
        "messages": workflows.message_log(run),
        "telemetry": normalize(langchain_telemetry(m for turn in run.turns for m in turn.messages)),
        **workflows.run_fields(run),
    }


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    per_test_timeout_s: int = task.TEST_TIMEOUT_S,
) -> dict:
    """Solve every instance and score it on its tests."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(
            lambda: solve(inst["problem"], starter_code=inst.get("starter_code") or None), fallback={"code": None}
        )
        code = out["code"]
        return task.record(
            inst,
            code,
            task.test_scores(code, inst["input_output"], per_test_timeout_s),
            raw=(out.get("raw") or "")[:2000],
            termination=out.get("termination"),
            rounds=out.get("rounds"),
            n_turns=out.get("n_turns"),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=f"{TOPOLOGY}/APPS")


def _canned_demo() -> None:
    for mode, problem, starter, tests in task.DEMOS:
        print(f"\n========== {mode} MODE ==========")
        out = solve(problem, starter_code=starter)
        task.print_demo_code(out["code"], tests)
        print(f"=== {out['n_turns']} turns, {out['termination']} ===")
        for m in out["messages"]:
            snippet = m["content"][:300].replace("\n", " ")
            print(f"  [{m['agent']} / {m['phase']} / round {m['round']}] {snippet}")


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description=f"{TOPOLOGY} APPS runner (LangGraph).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        predictions=DEFAULT_PREDICTIONS,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
