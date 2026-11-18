"""GPQA-Diamond under an extension topology (LangGraph): Fan-out/Fan-in or Iterative Refinement.

The topology's graph is :mod:`core.workflows`; every agent turn is a ReAct agent
with the role's prompt and tools (``configs/teams/gpqa.yaml``) on the question and
its shuffled options, followed by the turn's instruction. The answer is the option
letter of the final turn: the coordinator's synthesis, or the generator's latest
candidate. Executed with ``TOPOLOGY`` preset by
``topologies/<topology>/gpqa/langgraph_gpqa.py`` (:mod:`core.variant`).
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core import cli, prompts, settings, teams, workflows
from core.batch import attempt
from core.calculator import CALCULATOR_DOC, make_calculator
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import gpqa as task
from core.tasks.gpqa import extract_answer, load_instances
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_ai_thinking

TOPOLOGY = globals()["TOPOLOGY"]  # preset by the topologies/<topology>/gpqa entries (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET)
DEFAULT_PREDICTIONS = RESULTS_DIR / f"gpqa_{TOPOLOGY}_langgraph" / "predictions.jsonl"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
_RECURSION_LIMIT = TEAM.recursion_limit

TOOLS = {"calculator": tool(make_calculator(CALCULATOR_DOC))}


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_llm():
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


async def _call(question: str, choices: list[str], request: workflows.Request) -> workflows.Turn:
    """One turn: the role's ReAct agent on the question and the turn's instruction."""
    agent = create_react_agent(
        model=_build_llm(),
        tools=[TOOLS[name] for name in TEAM.tools(request.role)],
        prompt=_load_prompt(request.role),
    )
    user = f"{task.format_prompt(question, choices)}\n\n{request.instruction}"
    result = await agent.ainvoke({"messages": [("user", user)]}, config={"recursion_limit": _RECURSION_LIMIT})
    strip_ai_thinking(result["messages"])
    text = result["messages"][-1].content or ""
    artifact = extract_answer(text) if request.phase in workflows.ARTIFACT_PHASES else None
    return workflows.Turn(request, text, artifact=artifact, messages=result["messages"])


def solve(question: str, choices: list[str]) -> dict:
    """Run the team on one question.

    Returns ``{"answer", "raw", "messages", "telemetry", "termination", "rounds",
    "subtasks", "verdicts", "n_turns"}``: the final turn's letter and text, every
    turn as ``{source, agent, phase, round, content}``, token/call counts and how
    the run ended.
    """
    run = asyncio.run(workflows.run_team(TEAM, partial(_call, question, choices)))
    final = run.final.text if run.final else ""
    return {
        "answer": extract_answer(final),
        "raw": final,
        "messages": workflows.message_log(run),
        "telemetry": normalize(langchain_telemetry(m for turn in run.turns for m in turn.messages)),
        **workflows.run_fields(run),
    }


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["question"], inst["choices"]))
        return task.record(
            inst,
            out["answer"],
            raw=out.get("raw") or "",
            termination=out.get("termination"),
            rounds=out.get("rounds"),
            n_turns=out.get("n_turns"),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=f"{TOPOLOGY}/GPQA-Diamond")


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION, task.DEMO_CHOICES)
    task.print_demo_answer(out["answer"])
    print(f"=== {out['n_turns']} turns, {out['termination']} ===")
    for m in out["messages"]:
        snippet = m["content"][:300].replace("\n", " ")
        print(f"  [{m['agent']} / {m['phase']} / round {m['round']}] {snippet}")


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description=f"{TOPOLOGY} GPQA-Diamond runner (LangGraph).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        predictions=DEFAULT_PREDICTIONS,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
