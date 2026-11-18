# Iterative Refinement

A generator writes a candidate, two critics review it in parallel, and a verifier either accepts it or sends feedback back to the generator, for at most three rounds. It is one of the two extension topologies: LangGraph only, with one fixed team.
{ .lede }

--8<-- "diagrams/iterative_refinement.svg"

<div class="facts" markdown>
<div><span>Agents</span>Generator + 2 critics + verifier</div>
<div><span>Rounds</span>At most 3</div>
<div><span>Frameworks</span>LangGraph</div>
<div><span>Aggregation</span>Generator's latest candidate</div>
</div>

## How it works

1. **Generate.** In round 1 the generator gets the task and writes a candidate solution under its output contract.
2. **Review.** `critic_1` and `critic_2` read the candidate in parallel and give specific, actionable feedback: critic 1 on correctness and the task's requirements, critic 2 on edge cases, tool use and regression risks. They are read-only turns: a critic must not change the candidate or write a replacement.
3. **Verify.** The verifier reads the candidate and both reviews and replies with only `{"passed": true or false, "feedback": "..."}`. The verdict is its own judgment; it cannot see the hidden evaluation, and the benchmark score never enters the loop. A reply without that object, or with a `passed` that is not a boolean or a `feedback` that is not a string, counts as a fail.
4. **Revise.** On a pass the run ends. On a fail, the generator revises its previous candidate from the verifier's feedback and both reviews, then the critics and the verifier run again on the new candidate. Candidates and reviews are passed on in full, except on API-Bank and ToolHop, which hand reviews on as their other runners do (the last 1,200 characters of each, at most 5,000 in all).
5. **Stop.** After a pass (`termination` is `accepted`) or after round 3 (`max_rounds`). The submission is the generator's latest candidate, accepted or not, and the runner scores it without looking at the gold answer or the tests.

Each turn is one agent run on the task followed by the turn's instruction. On seven datasets it is a ReAct agent built with `create_react_agent` and the role's tools. On API-Bank it is one model call and on ToolHop one tool loop with the row's tools; there, agent `i` (generator 0, critics 1 and 2, verifier 3) samples with seed `i`.

| Datasets | Tools | Answer the runner scores |
| --- | --- | --- |
| GPQA | `calculator`, all four roles | the option letter |
| MATH | `calculator`, all four roles | the `\boxed{}` answer |
| HotpotQA | `wikipedia_search`, `wikipedia_page`, all four roles; the generator also gets the short-form rules | the `Answer:` line |
| LiveCodeBench, APPS | `python_exec`, all four roles | the last fenced Python block |
| BFCL | none; the function schemas are in the task message as text | the last fenced canonical JSON call list |
| SWE-bench | generator: `file_read`, `str_replace`, `list_dir`, `search_repo`, `shell_exec`; critics and verifier: `file_read`, `list_dir`, `search_repo` | `git diff HEAD` of the run's checkout |
| ToolHop | the row's tools | the `<answer>` tag |
| API-Bank | none; the dialogue and API descriptions are in the prompt | one bracketed API call |

On SWE-bench, the generator edits the run's checkout, and the candidate is the checkout's diff after its turn. The critics and the verifier read the same checkout with read-only tools only; the runner refuses to start a read-only turn whose role has an editing tool.

## Agent roles and seed prompts

Four roles per dataset, one agent each:

| Role | Job | Seed prompt |
| --- | --- | --- |
| `generator` | writes and revises the candidate; the final role | `configs/prompts/iterative_refinement/<dataset>/generator.txt` |
| `critic_1` | reviews correctness and the task's requirements | `configs/prompts/iterative_refinement/<dataset>/critic_1.txt` |
| `critic_2` | reviews edge cases, tool use and regression risks | `configs/prompts/iterative_refinement/<dataset>/critic_2.txt` |
| `verifier` | accepts the candidate or returns feedback | `configs/prompts/iterative_refinement/<dataset>/verifier.txt` |

The generator carries the protected [output contract](../evaluation/protocol.md#output-contracts). Role descriptions are under `iterative_refinement:` in [`configs/prompts/roles.yaml`](https://anonymous.4open.science/r/MAS-PromptBench-Codebase/configs/prompts/roles.yaml). The `iterative_refinement:` entry of the team spec, `configs/teams/<dataset>.yaml`, sets each role's tools, `max_rounds` (3) and the agents' `recursion_limit`; API-Bank and ToolHop list no named tools there, because their tools come with each instance.

## Implementation

- **Graph:** `iterative_refinement` in [`core/workflows.py`](https://anonymous.4open.science/r/MAS-PromptBench-Codebase/core/workflows.py) builds a `StateGraph` with a `generate` node, edges from it to `critic_1` and `critic_2`, a `verify` node that waits for both critics, and a conditional edge from `verify` back to `generate` or to `END`.
- **Team:** one fixed four-agent team (`FIXED_TEAMS` in `core/teams.py`). There are no team-size or communication-format variants; every message is freeform.
- **Dataset runners:** one shared runner per dataset, `topologies/workflows/<dataset>.py`, supplies the per-turn agent call for both extension topologies. The entry `topologies/iterative_refinement/<dataset>/langgraph_<dataset>.py` runs it with `TOPOLOGY="iterative_refinement"` preset (`core/variant.py`), so it owns its own hooks.
- **Records:** every record adds `termination` (`accepted` or `max_rounds`), `rounds` (1 to 3) and `n_turns`. The message log lists each turn with its `agent`, `phase` (`generate`, `critique`, `verify`, `revise`) and `round`; the BFCL, SWE-bench, ToolHop and API-Bank runners write it to `traces/`, and the ToolHop and API-Bank records also keep the `verdicts` and each turn's scored output (`turn_outputs`).

## Run it

```bash title="Smoke demo"
python -m topologies.iterative_refinement.math.langgraph_math
```

```bash title="Batch run on 50 LiveCodeBench problems"
python -m topologies.iterative_refinement.lcb.langgraph_lcb --batch --limit 50 \
  --out-dir results/topologies_baseline/iterative_refinement_lcb
```

The runners take the same command line as every topology runner, and the BFCL, SWE-bench, ToolHop and API-Bank runners have no demo. `scripts/run_topologies.sh` does not sweep them. See the [task pages](../tasks/index.md) for each dataset's options.

## Optimize it

The protocol topology is `iterative_refinement`. It exists only as the four-agent LangGraph team with freeform messages, so `--framework`, `--team-size` and `--communication` accept only those values. Its cells are outside the experiment grid and belong to the `new_topologies` [extension study](../evaluation/protocol.md#extension-studies), which runs GEPA, MIPRO, MAPRO and MASPO on all nine datasets:

```bash title="MASPO on Iterative Refinement · LiveCodeBench"
python -m optimizers.protocol.run --method maspo --dataset lcb \
  --topology iterative_refinement --model qwen --seed 0 --study new_topologies \
  --out runs/new_topologies/maspo/lcb/iterative_refinement/qwen/0
```

The optimizer tunes all four prompts. The bridge adapter is `IterativeRefinement<Dataset>Adapter` in `optimizers/bridge/adapters/module_<dataset>.py`. See [Run an Optimizer](../optimizers/running.md#run-an-extension-study).
