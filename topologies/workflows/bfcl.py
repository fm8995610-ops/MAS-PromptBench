"""BFCL under an extension topology (LangGraph): Fan-out/Fan-in or Iterative Refinement.

The topology's graph is :mod:`core.workflows`; every agent turn is a tool-less
ReAct agent (``configs/teams/bfcl.yaml``) with the role's prompt on the centralized
team's task message (the request, the function schemas and the canonical call
form), followed by the turn's instruction. As in the other multi-agent BFCL
runners, no agent gets the schemas as tools: the prediction is the last fenced
canonical JSON call list (:mod:`core.bfcl_calls`) of the final turn, the
coordinator's synthesis or the generator's latest candidate; the other turns
report, review or return a control message. Executed with ``TOPOLOGY`` preset by
``topologies/<topology>/bfcl/langgraph_bfcl.py`` (:mod:`core.variant`).
"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from pathlib import Path

from langgraph.prebuilt import create_react_agent

from core import prompts, settings, teams, workflows
from core.llm import chat_openai
from core.paths import RESULTS_DIR
from core.tasks import bfcl as task
from core.tasks.bfcl import AST_CATEGORIES, HF_DATASET, extract_canonical, load_instances  # noqa: F401
from core.telemetry import langchain_telemetry, normalize
from core.thinking import strip_ai_thinking

TOPOLOGY = globals()["TOPOLOGY"]  # preset by the topologies/<topology>/bfcl entries (core.variant)
TEAM = teams.spec(TOPOLOGY, task.DATASET)
DEFAULT_OUT_DIR = RESULTS_DIR / f"bfcl_{TOPOLOGY}_langgraph"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()
_RECURSION_LIMIT = TEAM.recursion_limit

_register_model_with_bfcl = task.register_model  # called again by callers that repoint MODEL_ID
_register_model_with_bfcl(MODEL_ID)


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_llm():
    return chat_openai(model=MODEL_ID, base_url=VLLM_BASE_URL)


def score_one(function_schemas: list[dict], model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    return task.score_one(function_schemas, model_output, ground_truth, category, MODEL_ID)


async def _call(prompt: str, request: workflows.Request) -> workflows.Turn:
    """One turn: the role's ReAct agent on the task message and the turn's instruction."""
    agent = create_react_agent(model=_build_llm(), tools=[], prompt=_load_prompt(request.role))
    user = f"{prompt}\n\n{request.instruction}"
    result = await agent.ainvoke({"messages": [("user", user)]}, config={"recursion_limit": _RECURSION_LIMIT})
    strip_ai_thinking(result["messages"])
    text = result["messages"][-1].content or ""
    artifact = extract_canonical(text) if request.phase in workflows.ARTIFACT_PHASES else None
    return workflows.Turn(request, text, artifact=artifact, messages=result["messages"])


def solve(instance: dict) -> dict:
    """Run the team on one instance.

    Returns ``{"model_output", "raw", "messages", "telemetry", "termination",
    "rounds", "subtasks", "verdicts", "n_turns"}``: the final turn's canonical
    calls (or []) and text, every turn as ``{source, agent, phase, round,
    content}``, token/call counts and how the run ended.
    """
    prompt = task.format_task(task.flatten_question(instance["question"]), task.render_schemas(instance["function"]))
    run = asyncio.run(workflows.run_team(TEAM, partial(_call, prompt)))
    final = run.final.text if run.final else ""
    return {
        "model_output": extract_canonical(final) or [],
        "raw": final,
        "messages": workflows.message_log(run),
        "telemetry": normalize(langchain_telemetry(m for turn in run.turns for m in turn.messages)),
        **workflows.run_fields(run),
    }


def _trace(messages: list[dict]) -> str:
    """Trace text of the turns, each under its agent, phase and round."""
    sections = ((f"{m['agent'].upper()} / {m['phase']} / round {m['round']}", m["content"]) for m in messages)
    return task.sections_trace(sections)


def run_one(instance: dict, ground_truth: dict, category: str, out_dir: Path) -> dict:
    """Solve and score one instance and write its turns to ``out_dir/traces/<id>.txt``."""
    summary: dict = {"id": instance["id"], "category": category}
    t0 = time.time()
    try:
        out = solve(instance)
    except Exception as e:
        task.solve_failed(summary, e)
        summary["solve_s"] = round(time.time() - t0, 1)
        return summary
    summary["solve_s"] = round(time.time() - t0, 1)
    summary["model_output"] = out["model_output"]
    summary["tool_calls"] = len(out["model_output"])
    summary.update(termination=out["termination"], rounds=out["rounds"], n_turns=out["n_turns"])
    summary.update(out["telemetry"])
    task.write_trace(out_dir, instance["id"], _trace(out["messages"]))
    return task.add_verdict(summary, score_one, instance, ground_truth, category)


def run_batch(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Score a slice of one subset; writes predictions.jsonl and results.jsonl afresh in ``out_dir``."""
    return task.evaluate(
        run_one, category, limit, offset, only, out_dir or DEFAULT_OUT_DIR, model_id=MODEL_ID, verbose=verbose
    )


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description=f"{TOPOLOGY} BFCL runner (LangGraph).",
        run_one=run_one,
        model_id=MODEL_ID,
        default_out_dir=DEFAULT_OUT_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
