"""The two extension topologies, Fan-out/Fan-in and Iterative Refinement, as LangGraph graphs.

``fan_out_fan_in``: one coordinator decomposes the task once into ``n_workers``
complementary subtasks (a JSON control message); the workers, who share one role
and one prompt, solve them in parallel without seeing each other's work; the same
coordinator synthesizes their reports into the final artifact. A decomposition
that is not ``n_workers`` distinct non-empty subtasks ends the run without an
artifact: a wrong answer the scorer sees, not an exception.

``iterative_refinement``: a generator writes a candidate; two critics review it in
parallel without changing it, one for correctness and requirements, one for edge
cases, tool use and regressions; a verifier reads the candidate and both reviews
and returns ``{"passed": bool, "feedback": str}``. On a fail the generator revises
the same candidate from the verifier's and the critics' feedback, for at most
``max_rounds`` rounds; a malformed verdict is a fail. The final artifact is the
generator's latest candidate. The verdict is the verifier's own judgment, never
the benchmark score.

A runner supplies ``call(request) -> Turn``: one agent turn with the dataset's
tools, the role's system prompt and the task followed by ``request.instruction``
(:func:`run_team` runs a team of :mod:`core.teams` with its topology's graph).
A turn's ``text`` is what other agents read; its ``artifact`` is the exact artifact
it produced and is never truncated: the candidate final answer, code, patch or call
in :data:`ARTIFACT_PHASES`, or a result a runner attaches to a worker's report
(e.g. the diff of a SWE-bench worker's own clone). ``render`` turns reports into
the context another agent reads: :func:`render_reports` (the default) passes them
on whole, while a runner may apply its dataset's handoff policy (the API-Bank and
ToolHop runners keep the tail of each report, as their other topologies do). The
candidate a critic or the verifier reviews is always shown whole (:func:`report`).
"""

from __future__ import annotations

import json
import operator
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from core.teams import TeamSpec

N_WORKERS = 3
MAX_ROUNDS = 3
# Phases whose turn carries a candidate final artifact (the others report or control the run).
ARTIFACT_PHASES = frozenset({"synthesize", "generate", "revise"})
CRITIC_FOCUS = {
    "critic_1": "correctness and the task's requirements",
    "critic_2": "edge cases, tool use and regression risks",
}

DECOMPOSE = (
    "You are the coordinator. This turn is the planning step: decompose the task above into exactly "
    "{n} distinct, complementary subtasks for {n} workers who share one role and work in parallel "
    "without seeing each other. Give each worker a different part of the work instead of the whole "
    "task. Your final-output contract applies to your synthesis turn, not to this one: reply with "
    'only this JSON object.\n{{"subtasks": [{slots}]}}'
)
SOLVE_SUBTASK = (
    "You are worker {index} of {n}. Solve only the subtask below and report your result, with the "
    "evidence or work that supports it, to the coordinator.\n\nSUBTASK:\n{subtask}"
)
SYNTHESIZE = (
    "You are the coordinator. Synthesize the workers' reports below into the final answer to the task: "
    "resolve inconsistencies between them and fill in missing pieces.\n\nWORKER REPORTS:\n{reports}"
)
GENERATE = "Produce a candidate solution to the task."
REVISE = (
    "Revise your candidate using the feedback below: keep what is correct and fix what is not.\n\n"
    "YOUR PREVIOUS CANDIDATE:\n{candidate}\n\nFEEDBACK:\n{feedback}"
)
CRITIQUE = (
    "Review the candidate below for {focus}. Give specific, actionable feedback; do not change the "
    "candidate or write a replacement for it.\n\nCANDIDATE:\n{candidate}"
)
VERIFY = (
    "Decide whether the candidate below is ready to submit, using the two reviews. This is your own "
    "judgment; you cannot see the hidden evaluation. Reply with only this JSON object:\n"
    '{{"passed": true or false, "feedback": "<what the generator must change, or why the candidate passes>"}}'
    "\n\nCANDIDATE:\n{candidate}\n\nREVIEWS:\n{reviews}"
)
MALFORMED_VERDICT = 'The verifier reply was not the required JSON object {"passed": ..., "feedback": ...}.'


@dataclass(frozen=True)
class Request:
    """One agent turn the topology asks a runner to execute."""

    role: str  # prompt role: coordinator, worker, generator, critic_1, critic_2, verifier
    agent: str  # agent ID: coordinator, worker_<i>, generator, critic_<i>, verifier
    phase: str  # decompose, solve_subtask, synthesize, generate, revise, critique, verify
    round: int
    instruction: str  # the phase's message, sent after the task itself
    read_only: bool = False  # critics and the verifier must not change the candidate


@dataclass
class Turn:
    """An executed turn: the report other agents read, the exact artifact and the framework trace."""

    request: Request
    text: str
    artifact: Any = None
    messages: list = field(default_factory=list)
    telemetry: dict = field(default_factory=dict)


@dataclass
class Run:
    """One run of a topology: every turn in order, the final turn and how the run ended."""

    turns: list[Turn]
    final: Turn | None
    rounds: int
    termination: str  # completed, invalid_decomposition, accepted, max_rounds
    subtasks: list[str] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)


Call = Callable[[Request], Awaitable[Turn]]
Render = Callable[[Sequence[Turn]], str]


def report(turn: Turn) -> str:
    """A turn as another agent reads it: its text, plus its artifact when the text does not contain it."""
    artifact = turn.artifact
    if artifact is None or artifact == "":
        return turn.text
    shown = artifact if isinstance(artifact, str) else json.dumps(artifact, ensure_ascii=False, default=str)
    return turn.text if shown in turn.text else f"{turn.text}\n\nARTIFACT (exact):\n{shown}"


def render_reports(turns: Sequence[Turn]) -> str:
    """Each turn's :func:`report` under its agent ID, in order (no rewriting or truncation)."""
    return "\n\n".join(f"{turn.request.agent}:\n{report(turn)}" for turn in turns)


def control_json(text: str, key: str) -> dict | None:
    """The first JSON object in ``text`` that has ``key`` (a control message may carry extra prose)."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and key in value:
            return value
        start = text.find("{", start + 1)
    return None


def parse_subtasks(text: str, n: int) -> list[str] | None:
    """``n`` distinct non-empty subtasks from a decomposition reply, else None."""
    value = control_json(text, "subtasks")
    subtasks = value.get("subtasks") if value else None
    if not isinstance(subtasks, list) or len(subtasks) != n:
        return None
    if not all(isinstance(item, str) and item.strip() for item in subtasks):
        return None
    if len({" ".join(item.lower().split()) for item in subtasks}) != n:
        return None
    return [item.strip() for item in subtasks]


def parse_verdict(text: str) -> dict:
    """``{"passed", "feedback", "well_formed"}`` from a verifier reply; a malformed reply does not pass."""
    value = control_json(text, "passed")
    if value is None or type(value.get("passed")) is not bool or not isinstance(value.get("feedback"), str):
        return {"passed": False, "feedback": MALFORMED_VERDICT, "well_formed": False}
    return {"passed": value["passed"], "feedback": value["feedback"], "well_formed": True}


# Fan-out/Fan-in
class FanOutState(TypedDict, total=False):
    turns: Annotated[list[Turn], operator.add]
    subtasks: list[str]
    final: Turn


class WorkerInput(TypedDict):
    index: int
    subtask: str


async def fan_out_fan_in(call: Call, *, n_workers: int = N_WORKERS, render: Render = render_reports) -> Run:
    """Decompose once, solve the subtasks in parallel, synthesize; see the module docstring."""

    async def decompose(state: FanOutState) -> dict:
        slots = ", ".join(f'"<subtask {i}>"' for i in range(1, n_workers + 1))
        request = Request("coordinator", "coordinator", "decompose", 1, DECOMPOSE.format(n=n_workers, slots=slots))
        turn = await call(request)
        return {"turns": [turn], "subtasks": parse_subtasks(turn.text, n_workers) or []}

    def dispatch(state: FanOutState) -> list[Send] | str:
        subtasks = state.get("subtasks") or []
        if not subtasks:
            return END
        return [Send("worker", {"index": i, "subtask": subtask}) for i, subtask in enumerate(subtasks, 1)]

    async def worker(state: WorkerInput) -> dict:
        index = state["index"]
        instruction = SOLVE_SUBTASK.format(index=index, n=n_workers, subtask=state["subtask"])
        return {"turns": [await call(Request("worker", f"worker_{index}", "solve_subtask", 1, instruction))]}

    async def synthesize(state: FanOutState) -> dict:
        workers = sorted(
            (turn for turn in state["turns"] if turn.request.phase == "solve_subtask"), key=lambda t: t.request.agent
        )
        instruction = SYNTHESIZE.format(reports=render(workers))
        turn = await call(Request("coordinator", "coordinator", "synthesize", 1, instruction))
        return {"turns": [turn], "final": turn}

    graph = StateGraph(FanOutState)
    graph.add_node("decompose", decompose)
    graph.add_node("worker", worker)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "decompose")
    graph.add_conditional_edges("decompose", dispatch, ["worker", END])
    graph.add_edge("worker", "synthesize")
    graph.add_edge("synthesize", END)
    state = await graph.compile().ainvoke({"turns": []})
    final = state.get("final")
    return Run(
        turns=list(state["turns"]),
        final=final,
        rounds=1,
        termination="completed" if final is not None else "invalid_decomposition",
        subtasks=list(state.get("subtasks") or []),
    )


# Iterative Refinement
class RefinementState(TypedDict, total=False):
    turns: Annotated[list[Turn], operator.add]
    verdicts: Annotated[list[dict], operator.add]
    round: int
    candidate: Turn
    feedback: str
    passed: bool


async def iterative_refinement(call: Call, *, max_rounds: int = MAX_ROUNDS, render: Render = render_reports) -> Run:
    """Generate, review in parallel, verify; revise until a pass or ``max_rounds``; see the module docstring."""

    async def generate(state: RefinementState) -> dict:
        round_index = state.get("round", 0) + 1
        if round_index == 1:
            request = Request("generator", "generator", "generate", 1, GENERATE)
        else:
            instruction = REVISE.format(candidate=report(state["candidate"]), feedback=state["feedback"])
            request = Request("generator", "generator", "revise", round_index, instruction)
        turn = await call(request)
        return {"turns": [turn], "candidate": turn, "round": round_index}

    def critic(role: str):
        async def review(state: RefinementState) -> dict:
            instruction = CRITIQUE.format(focus=CRITIC_FOCUS[role], candidate=report(state["candidate"]))
            request = Request(role, role, "critique", state["round"], instruction, read_only=True)
            return {"turns": [await call(request)]}

        return review

    async def verify(state: RefinementState) -> dict:
        reviews = _round_turns(state["turns"], state["round"], "critique")
        instruction = VERIFY.format(candidate=report(state["candidate"]), reviews=render(reviews))
        turn = await call(Request("verifier", "verifier", "verify", state["round"], instruction, read_only=True))
        verdict = {"round": state["round"], **parse_verdict(turn.text)}
        # The critics' reviews travel with the verdict, so short or malformed verdicts keep their feedback.
        feedback = f"VERIFIER:\n{verdict['feedback']}\n\nREVIEWS:\n{render(reviews)}"
        return {"turns": [turn], "verdicts": [verdict], "passed": verdict["passed"], "feedback": feedback}

    def next_step(state: RefinementState) -> str:
        return END if state["passed"] or state["round"] >= max_rounds else "generate"

    graph = StateGraph(RefinementState)
    graph.add_node("generate", generate)
    for role in CRITIC_FOCUS:
        graph.add_node(role, critic(role))
        graph.add_edge("generate", role)
    graph.add_node("verify", verify)
    graph.add_edge(START, "generate")
    graph.add_edge(list(CRITIC_FOCUS), "verify")
    graph.add_conditional_edges("verify", next_step, ["generate", END])
    state = await graph.compile().ainvoke({"turns": [], "verdicts": []}, config={"recursion_limit": 8 * max_rounds})
    return Run(
        turns=list(state["turns"]),
        final=state["candidate"],
        rounds=state["round"],
        termination="accepted" if state["passed"] else "max_rounds",
        verdicts=list(state["verdicts"]),
    )


def _round_turns(turns: Sequence[Turn], round_index: int, phase: str) -> list[Turn]:
    """The turns of ``phase`` in round ``round_index``, ordered by agent."""
    selected = [t for t in turns if t.request.round == round_index and t.request.phase == phase]
    return sorted(selected, key=lambda t: t.request.agent)


def message_log(run: Run) -> list[dict]:
    """Every turn as ``{source, agent, phase, round, content}``; ``source`` is the prompt role."""
    return [
        {
            "source": turn.request.role,
            "agent": turn.request.agent,
            "phase": turn.request.phase,
            "round": turn.request.round,
            "content": turn.text,
        }
        for turn in run.turns
    ]


def run_fields(run: Run) -> dict:
    """The topology-level fields of a run's output and record."""
    return {
        "termination": run.termination,
        "rounds": run.rounds,
        "subtasks": run.subtasks,
        "verdicts": run.verdicts,
        "n_turns": len(run.turns),
    }


async def run_team(team: TeamSpec, call: Call, *, render: Render = render_reports) -> Run:
    """Run ``team`` (a fixed team of :mod:`core.teams`) with its topology's graph."""
    if team.topology == "fan_out_fan_in":
        return await fan_out_fan_in(call, n_workers=team.n_workers, render=render)
    if team.topology == "iterative_refinement":
        return await iterative_refinement(call, max_rounds=team.max_rounds, render=render)
    raise ValueError(f"{team.topology!r} is not an extension topology")
