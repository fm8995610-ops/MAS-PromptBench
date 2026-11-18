"""HotpotQA under an extension topology (LangGraph): Fan-out/Fan-in or Iterative Refinement.

The topology's graph is :mod:`core.workflows`; every agent turn is a ReAct agent
with the role's prompt and Wikipedia tools (``configs/teams/hotpotqa.yaml``) on the
question, followed by the turn's instruction. The final role (the coordinator or
the generator) also gets the short-form ``Answer:`` rules; the answer is the
``Answer:`` line of the final turn: the coordinator's synthesis, or the generator's
latest candidate. Executed with ``TOPOLOGY`` preset by
``topologies/<topology>/hotpotqa/langgraph_hotpotqa.py`` (:mod:`core.variant`).
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core import cli, prompts, settings, teams, workflows
from core.batch import attempt
from core.llm import chat_openai
from core.output_contracts import EXTENSION_FINAL_ROLES
from core.paths import RESULTS_DIR
from core.tasks import hotpotqa as task
from core.tasks.hotpotqa import (  # noqa: F401  (runner API)
    exact_match_score,
    extract_answer,
    f1_score,
    load_instances,
    normalize_answer,
)
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_ai_thinking

TOPOLOGY = globals()["TOPOLOGY"]  # preset by the topologies/<topology>/hotpotqa entries (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET)
DEFAULT_PREDICTIONS = RESULTS_DIR / f"hotpotqa_{TOPOLOGY}_langgraph" / "predictions.jsonl"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
_RECURSION_LIMIT = TEAM.recursion_limit

# The short-form rules of the single, independent and decentralized runners, for the final role only (as
# the centralized manager gets them): the other roles report, review or return a JSON verdict. They are
# appended per turn, not in _load_prompt (which the optimizer adapters replace), and are not named
# _OUTPUT_FORMAT_NUDGE (which the adapters append to every role).
FINAL_ROLES = EXTENSION_FINAL_ROLES[TOPOLOGY]
_FINAL_ROLE_NUDGE = task.OUTPUT_FORMAT_NUDGE

wikipedia_search = tool(task.make_wikipedia_search(task.SEARCH_DOC))
wikipedia_page = tool(task.make_wikipedia_page(task.PAGE_DOC))
TOOLS = {"wikipedia_search": wikipedia_search, "wikipedia_page": wikipedia_page}


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_llm():
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _system_prompt(role: str) -> str:
    """The role's prompt, plus the short-form rules for the final role."""
    return _load_prompt(role) + (_FINAL_ROLE_NUDGE if role in FINAL_ROLES else "")


async def _call(question: str, request: workflows.Request) -> workflows.Turn:
    """One turn: the role's ReAct agent on the question and the turn's instruction."""
    agent = create_react_agent(
        model=_build_llm(),
        tools=[TOOLS[name] for name in TEAM.tools(request.role)],
        prompt=_system_prompt(request.role),
    )
    user = f"{task.format_prompt(question)}\n\n{request.instruction}"
    result = await agent.ainvoke({"messages": [("user", user)]}, config={"recursion_limit": _RECURSION_LIMIT})
    strip_ai_thinking(result["messages"])
    text = result["messages"][-1].content or ""
    artifact = extract_answer(text) if request.phase in workflows.ARTIFACT_PHASES else None
    return workflows.Turn(request, text, artifact=artifact, messages=result["messages"])


def solve(question: str) -> dict:
    """Run the team on one question.

    Returns ``{"answer", "raw", "messages", "telemetry", "termination", "rounds",
    "subtasks", "verdicts", "n_turns"}``: the final turn's short-form answer and
    text, every turn as ``{source, agent, phase, round, content}``, token/call
    counts and how the run ended.
    """
    run = asyncio.run(workflows.run_team(TEAM, partial(_call, question)))
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
        out, latency_s, error = attempt(lambda: solve(inst["question"]))
        return task.record(
            inst,
            out["answer"],
            **task.meta(inst),
            raw=out.get("raw") or "",
            termination=out.get("termination"),
            rounds=out.get("rounds"),
            n_turns=out.get("n_turns"),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(
        instances, row, out_path=out_path, verbose=verbose, label=f"{TOPOLOGY}/HotpotQA", marks=task.ASCII_MARKS
    )


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION)
    task.print_demo_answer(out["answer"])
    print(f"=== {out['n_turns']} turns, {out['termination']} ===")
    for m in out["messages"]:
        snippet = m["content"][:300].replace("\n", " ")
        print(f"  [{m['agent']} / {m['phase']} / round {m['round']}] {snippet}")


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description=f"{TOPOLOGY} HotpotQA runner (LangGraph).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        predictions=DEFAULT_PREDICTIONS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
