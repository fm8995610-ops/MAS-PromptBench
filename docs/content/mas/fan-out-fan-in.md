# Fan-out/Fan-in

A coordinator splits the task once into three complementary subtasks, three workers solve them in parallel without seeing each other's work, and the same coordinator synthesizes their reports into the answer. It is one of the two extension topologies: LangGraph only, with one fixed team.
{ .lede }

--8<-- "diagrams/fan_out_fan_in.svg"

<div class="facts" markdown>
<div><span>Agents</span>1 coordinator + 3 workers</div>
<div><span>Rounds</span>1</div>
<div><span>Frameworks</span>LangGraph</div>
<div><span>Aggregation</span>Coordinator synthesizes</div>
</div>

## How it works

1. **Decompose.** The coordinator gets the task and a planning instruction, and replies with only a JSON object, `{"subtasks": ["...", "...", "..."]}`: three distinct, complementary subtasks for workers who work in parallel. Its final-output contract applies to the synthesis turn, not to this one.
2. **Check the plan.** The runner reads the first JSON object with a `subtasks` key. If it is not exactly three distinct, non-empty strings (compared ignoring case and whitespace), the run ends with no synthesis and no answer. The record's `termination` is `invalid_decomposition`, and the scorer sees a wrong answer, not an error.
3. **Fan out.** A conditional edge returns one LangGraph `Send` per subtask (`worker_1` to `worker_3`). Each worker gets the task and only its own subtask, solves it with the dataset's tools and reports its result with the supporting work. Workers never see each other's subtasks or reports.
4. **Fan in.** The coordinator, with the same prompt as in step 1, reads the three reports in worker order and synthesizes the final answer: it resolves inconsistencies and fills in missing pieces. A report is the worker's reply plus its exact artifact when the reply does not contain it, such as a SWE-bench worker's diff, and is passed on in full. API-Bank and ToolHop are the exception: they hand reports on as their other runners do, the last 1,200 characters of each and at most 5,000 in all.
5. The runner extracts and scores the answer of the synthesis turn, without looking at the gold answer or the tests.

Each turn is one agent run on the task followed by the turn's instruction. On seven datasets it is a ReAct agent built with `create_react_agent` and the role's tools. On API-Bank it is one model call and on ToolHop one tool loop with the row's tools; there, agent `i` (coordinator 0, workers 1 to 3) samples with seed `i`.

| Datasets | Tools (both roles) | Answer the runner scores |
| --- | --- | --- |
| GPQA | `calculator` | the option letter |
| MATH | `calculator` | the `\boxed{}` answer |
| HotpotQA | `wikipedia_search`, `wikipedia_page` | the `Answer:` line; the coordinator also gets the short-form rules |
| LiveCodeBench, APPS | `python_exec` | the last fenced Python block |
| BFCL | none; the function schemas are in the task message as text | the last fenced canonical JSON call list |
| SWE-bench | `file_read`, `str_replace`, `list_dir`, `search_repo`, `shell_exec` | `git diff HEAD` of the run's checkout |
| ToolHop | the row's tools | the `<answer>` tag |
| API-Bank | none; the dialogue and API descriptions are in the prompt | one bracketed API call |

On SWE-bench, each worker works in its own clone of the run's checkout, made at the start of its turn from the checked-out commit and sandboxed with the instance's image. It reports that clone's diff, and the clones are removed after the run. The coordinator plans and synthesizes in the run's checkout, where it applies the combined fix.

## Agent roles and seed prompts

Two roles per dataset:

| Role | Agents | Seed prompt |
| --- | --- | --- |
| `coordinator` | 1; it writes the plan and the synthesis | `configs/prompts/fan_out_fan_in/<dataset>/coordinator.txt` |
| `worker` | 3, sharing one prompt | `configs/prompts/fan_out_fan_in/<dataset>/worker.txt` |

The coordinator is the final role and carries the protected [output contract](../evaluation/protocol.md#output-contracts). Role descriptions are under `fan_out_fan_in:` in [`configs/prompts/roles.yaml`](https://anonymous.4open.science/r/MAS-PromptBench-Codebase/configs/prompts/roles.yaml). Each role's tools and the agents' `recursion_limit` come from the `fan_out_fan_in:` entry of the team spec, `configs/teams/<dataset>.yaml`; API-Bank and ToolHop list no named tools there, because their tools come with each instance.

## Implementation

- **Graph:** `fan_out_fan_in` in [`core/workflows.py`](https://anonymous.4open.science/r/MAS-PromptBench-Codebase/core/workflows.py) builds a `StateGraph` with a `decompose` node, a conditional edge that returns the `Send`s (or `END` after an invalid plan), a `worker` node and a `synthesize` node. Turns merge through an `operator.add` reducer.
- **Team:** one fixed four-agent team (`FIXED_TEAMS` in `core/teams.py`, 3 workers). There are no team-size or communication-format variants; every message is freeform.
- **Dataset runners:** one shared runner per dataset, `topologies/workflows/<dataset>.py`, supplies the per-turn agent call for both extension topologies. The entry `topologies/fan_out_fan_in/<dataset>/langgraph_<dataset>.py` runs it with `TOPOLOGY="fan_out_fan_in"` preset (`core/variant.py`), so it owns its own hooks.
- **Records:** every record adds `termination` (`completed` or `invalid_decomposition`), `rounds` (1) and `n_turns`. The message log lists each turn with its `agent`, `phase` (`decompose`, `solve_subtask`, `synthesize`) and `round`; the BFCL, SWE-bench, ToolHop and API-Bank runners write it to `traces/`, and the ToolHop and API-Bank records also keep the `subtasks` and each turn's scored output (`turn_outputs`).

## Run it

```bash title="Smoke demo"
python -m topologies.fan_out_fan_in.math.langgraph_math
```

```bash title="Batch run on 100 MATH problems"
python -m topologies.fan_out_fan_in.math.langgraph_math --batch --limit 100 \
  --out-dir results/topologies_baseline/fan_out_fan_in_math
```

The runners take the same command line as every topology runner, and the BFCL, SWE-bench, ToolHop and API-Bank runners have no demo. `scripts/run_topologies.sh` does not sweep them. See the [task pages](../tasks/index.md) for each dataset's options.

## Optimize it

The protocol topology is `fan_out_fan_in`. It exists only as the four-agent LangGraph team with freeform messages, so `--framework`, `--team-size` and `--communication` accept only those values. Its cells are outside the experiment grid and belong to the `new_topologies` [extension study](../evaluation/protocol.md#extension-studies), which runs GEPA, MIPRO, MAPRO and MASPO on all nine datasets:

```bash title="GEPA on Fan-out/Fan-in · MATH"
python -m optimizers.protocol.run --method gepa --dataset math \
  --topology fan_out_fan_in --model qwen --seed 0 --study new_topologies \
  --out runs/new_topologies/gepa/math/fan_out_fan_in/qwen/0
```

The optimizer tunes two prompts: the coordinator's and the one worker prompt the three workers share. The bridge adapter is `FanOutFanIn<Dataset>Adapter` in `optimizers/bridge/adapters/module_<dataset>.py`. See [Run an Optimizer](../optimizers/running.md#run-an-extension-study).
