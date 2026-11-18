# Run Protocol

**`mas-promptbench`** is the one run protocol for every prompt optimizer. A **job** is `(method, cell, optimizer seed ∈ {0, 1, 2})`; a **cell** is `(task, topology, framework, communication, team size, task model)`, plus the seed-prompt set when it is not P1.

Part of [optimizers/](../README.md). Every rollout runs through the [real-runner bridge](../bridge/README.md); the seed prompts in `configs/prompts/` are **read-only**.

## Overview

1. **Optimize** — fixed ordered train/validation rows, the frozen seed bundle and **600 usable full-MAS rollouts** (temperature 0.2, top-p 0.9, ≤ 32,768 tokens).
2. **Final validation** (uncharged, greedy) — seed and incumbent on the full validation split with paired per-item request seeds. The incumbent is deployed only if **strictly better**; otherwise (tie, regression, unusable records, infrastructure-invalid run) the seed stays. Sealed in `selection.json`.
3. **Test** (uncharged, greedy) — after the lock, seed and deployed bundles on the held-out test split, same paired seeds.

Request seeds are offset by 0 / 1000 / 2000 per optimizer seed. `aggregate` pairs seeds 0–2 per cell: per-seed baseline / deployed / Δ, mean and std of Δ, seed-stratified paired bootstrap CI, per-seed exact McNemar, Holm within a family. The reflection model is `Qwen/Qwen3.5-122B-A10B-FP8` (thinking on, ≤ 48,000 tokens, temperature and top-p 1.0 unless the method sets them).

### Modules

| Module | Contents |
|---|---|
| `run`, `aggregate` | job CLI; seed aggregation CLI |
| `config`, `settings`, `cells`, `studies` | frozen constants; environment; experiment grid; extension studies |
| `runner`, `adapter_output` | `ProtocolRunner`; adapter output → record fields |
| `budget`, `seeding` | `BudgetLedger`; request and paired evaluation seeds |
| `schema`, `errors` | `CellSpec`, `PromptBundle`, `RunRecord`, `OptimizerResult`, `StopReason`; all exceptions |
| `rollouts` | runner helpers for the text methods |
| `session`, `journal` | `RunnerSession` for methods with their own loop; `LifecycleJournal` |
| `reflection` | `ReflectionClient`, shared by every method |
| `evaluation`, `selection`, `learning_curve`, `reporting`, `artifacts` | final validation and test; deployment; learning curve; paired statistics; artifact storage |
| `methods/` | registry (`METHODS`, `build_optimizer`), `identity`, DSPy plumbing for GEPA and MIPRO |

---

## How it works

### Execution and charging (`runner.py`)

A rollout installs the bundle's role prompts (with MIPRO's demos) in the cell's bridge adapter, sets the request seed, model and phase decoding, and scores one `adapter.run_example()` with the dataset `metric`.

- **Usable** — `success` or `semantic_failure` (incl. wrong or malformed answers): charged.
- **Infrastructure failure** — an exception before a scored observation (connection, timeout, 5xx, BadRequest), transport-error text, no model call, or a scorer exception: never charged, retried twice with the same seed, then returned unusable.
- Batches reserve budget before dispatch, so a job never overshoots.
- Rollouts are serialized per process; run jobs in parallel as processes.

### Scoring

Every rollout in every phase is scored 0 or 1 by `optimizers.bridge.datasets.<dataset>.metric`; a split's score is the mean. **LiveCodeBench, APPS and SWE-bench use cheaper checks** than the topology runners report:

| Dataset | Protocol metric | Topology runners |
|---|---|---|
| `gpqa` | option letter equals the gold letter (`accuracy`) | same |
| `hotpotqa` | official exact match (`exact_match`) | exact match and token F1 |
| `math` | Hendrycks `is_equiv` of the last `\boxed{}` (`accuracy`) | same |
| `bfcl` | `bfcl_eval` AST checker (`accuracy`) | same |
| `apibank` | the runners' API-Bank replay check (`accuracy`) | same |
| `toolhop` | ToolHop's matcher on the final answer, else on the selected agent's last tool result (`accuracy`) | same |
| `lcb` | passes the **first 3** private tests (`pass_at_1`) | all private tests |
| `apps` | passes the **first 3** tests (`pass_at_1`) | the first 20 tests |
| `swe` | **structural check only**: a unified diff (header and ≥ 1 changed line); not applied, no tests run (`resolved_rate`) | every `FAIL_TO_PASS` and `PASS_TO_PASS` test passes in the instance's SWE-bench image |

The bridge's APPS loader keeps only `interview` problems (all APPS evaluation ids are `interview`).

---

## Usage

### Run a job

```bash
# from the repository root; serve the task and reflection models first (see models/)
export TASK_ENDPOINTS=http://localhost:8000/v1             # or --task-endpoints, or VLLM_BASE_URL
export REFLECTION_MODEL_BASE_URL=http://localhost:8200/v1  # reflection model (the default)

python -m optimizers.protocol.run --method gepa --dataset hotpotqa --topology sequential \
    --model qwen --seed 0 --out runs/gepa/hotpotqa/sequential/qwen/0
python -m optimizers.protocol.aggregate runs/ --out runs/summary.json
```

- `--topology` is a base topology (refined by `--framework`, `--team-size`, `--communication`), an extension topology (`fan_out_fan_in`, `iterative_refinement`: LangGraph, 4 agents, freeform only) or a registry key such as `sequential_crewai`; `--model` is `qwen` (`Qwen/Qwen3.5-9B`) or `llama` (`meta-llama/Llama-3.1-8B-Instruct`).
- `--prompt-set P2` starts the seed bundle from the frozen second set (`configs/prompt_sets/p2/`, [`core/prompt_sets.py`](../../core/prompt_sets.py)); default `P1`, `configs/prompts/`.
- Cells outside the grid (`cells.py`: 612 configurations, 1,836 jobs) need `--allow-any-cell`, unless `--study` names the [extension study](#extension-studies) they belong to; `identity` runs on any grid (or named study) runtime condition.
- **Non-conformant** jobs — `--budget` below 600 (smoke runs), an off-grid cell, or another `REFLECTION_MODEL_ID` — are skipped by `aggregate` unless `--include-nonconformant`.
- `--phase optimize|validate|test` runs one phase from saved artifacts; `all` (default) runs what is missing. An interrupted optimization never restarts silently (use a fresh `--out`); evaluations resume per item.
- An Agents SDK job (`decentralized_openai_agents`) restarts itself with the SDK's isolated install first on `PYTHONPATH`; without a usable SDK it exits with status 2.
- Logs go to stderr; stdout carries only the job's one-line JSON summary.

Every option: [Command-Line Flags](../../docs/content/reference/cli.md#run-protocol); every environment variable: [Environment Variables](../../docs/content/reference/environment.md#run-protocol). Artifacts under `--out` (JSON, no absolute paths or host names): `job.json`, `optimization/`, `optimization.json`, `evaluations/`, `selection.json`, `test.json`, `result.json`.

### Extension studies

`studies.py` defines four studies beyond the grid, all LangGraph with the Qwen task model and seeds 0–2:

| Study | Methods | Tasks | Runtimes | Configurations | Jobs |
|---|---|---|---|---:|---:|
| `new_topologies` | GEPA, MIPRO, MAPRO, MASPO | all 9 | `fan_out_fan_in`, `iterative_refinement` | 72 | 216 |
| `prompt_set2` | GEPA, MIPRO, MAPRO, MASPO | `hotpotqa`, `lcb`, `bfcl` | 4 multi-agent topologies, r = 4, seeds from P2 | 48 | 144 |
| `protocol_extension` | GEPA, MIPRO, MASPO | `math`, `swe`, `apibank` | 4 multi-agent topologies × 3 formats, r = 4 | 108 (36 in the grid) | 324 |
| `team_size_extension` | GEPA, MIPRO, MASPO | `math`, `swe`, `apibank` | 4 multi-agent topologies, freeform, r ∈ {4, 8, 10} | 108 (36 in the grid) | 324 |

- `run --study <name>` admits the study's cells; such jobs are conformant. A study cell that is also a grid cell (the freeform r = 4 cells of the last two studies) is that grid cell, with its source tables, so its job and results are shared.
- Study-only jobs record `study` in `job.json` and `result.json`; a P2 cell records `prompt_set` in `grid_cell`. `CellSpec.prompt_set` enters the job identity only when it is not P1, so P1 `cell_id`s are unchanged; `study` never does.
- `aggregate` skips study-only jobs unless `--study <name>`, which reports that study, shared grid cells included. `--group-by task,communication` (any fields of `method`, `task`, `topology`, `framework`, `communication`, `team_size`, `task_model`, `prompt_set`) adds one summary per group: per seed the mean over the group's cells, then the mean and sample std of the three seed deltas. P2 cells aggregate apart from P1 cells.
- The studies do not add to the grid (`build_grid`, `EXPECTED_COUNTS`); `EXPECTED_STUDY_COUNTS` pins their sizes.

```bash
python -m optimizers.protocol.studies --counts                                         # sizes of every study
python -m optimizers.protocol.studies new_topologies --out-root runs/new_topologies     # one run command per job
python -m optimizers.protocol.aggregate runs/ --study protocol_extension --group-by task,communication
```

### Smoke budgets

Methods charge rollouts before proposing anything, so a small `--budget` can end a job before any search. The smallest budget that scores one proposed candidate (train 150, validation 50; GPQA train 48, SWE-bench train 146):

| Method | Charged | Smallest budget | With less |
|---|---|---|---|
| `identity` | nothing | 1 | – |
| `gepa` | seed on validation (50); per iteration 6 (parent and candidate on 3 train rows), +50 on validation if the candidate wins | 56 (106 to validate a winner) | `rollout_budget_spent`, seed kept |
| `mipro` | demo bootstrapping on train (until ≥ 5 succeed, ≤ 2 × train), seed on validation (50); 50 per trial (3 trials) | bootstrap + 100: 105 to 400 (GPQA 196, SWE-bench 392) | unrun rows fail; `rollout_budget_spent` |
| `mamut_gepa` | as `gepa` | 56 (106) | `metric_call_cap`, seed kept |
| `mapro` | seed on all of train; one train pass per new MAP assignment (probes and judges uncharged) | 2 × train: 300 (GPQA 96, SWE-bench 292) | `budget` after the seed pass; `NativeIntegrationError` below one pass |
| `hivemind` | every coalition on 5 train rows (16 coalitions: 80; centralized 8: 40); 10 per cycle (current and candidate on 5 validation rows), started only if it fits, batches shrinking to 1 row | 18 (centralized 10) with 1-row batches; 90 (50) at full size | no cycle, `budget`, seed kept |
| `maspo` | seed and its 2 candidates on 10 train rows (30); ≤ 60 per step, ≤ 30 more on a role revisit | 20 (30 for both candidates) | unrun rows score 0; `budget` |
| `maspob` | `min(5, max(1, B // 5 - 1))` warm-up pulls of 5 train rows; 5 per LinUCB pull | 10 for one LinUCB pull (30 after 5 warm-ups) | warm-up only; `rollout_budget_spent` |
| `tavo` | seed on `min(50, max(3, (B - 30) // 6))` validation rows; per attempt 6 train rows plus that batch | 12 | `budget_before_outer_round`, seed kept; fails below 3 |

### Settings

Each method's knobs are one frozen dataclass; only MAPRO and TAVO also read environment variables. The protocol's and the bridge's variables are in [Environment Variables](../../docs/content/reference/environment.md) ([run protocol](../../docs/content/reference/environment.md#run-protocol), [optimizer bridge](../../docs/content/reference/environment.md#optimizer-bridge)).

| Knob | Default |
|---|---|
| `gepa/integration.py: GEPAPolicy` | `max_full_evals=5`, minibatch 3, Pareto selection, round-robin components, merge (≤ 5), plateau patience 3, seed 0 |
| `mipro/integration.py: MIPROPolicy` | 3 candidates, 3 trials, ≤ 4 bootstrapped / 0 labeled demos, no minibatch, seed 9, all proposer hints on |
| `hivemind/regime.py: HiveMindSettings` | coalition batch 5, acceptance batch 5, ≤ 40 coalitions, manager every 3rd cycle, ≤ 6 lessons, fail below 0.5, reflection temperature 0.7, `max_cycles=0` (until B) |
| `mamut_gepa/integration.py: MAMUTGEPASettings` | `max_metric_calls = min(600, B)`, Pareto, round-robin, merge, minibatch 3, reflection temperature 0.7 |
| `mapro/regime.py: MAPROSettings` | 5 candidates per role, ≤ 8 rounds, patience 3, scoring batch 3, feedback 3, 12 threads |
| `MAPRO_LISTWISE` | `0`; `1` = paper listwise scoring (refused by the protocol) |
| `MAPRO_PAPER_ANCHOR` | `0`; `1` = latest-selection anchor (refused by the protocol) |
| `MAPRO_INIT_MAX_TOKENS` | `1024` (pool-initialization rewrite cap) |
| `MAPRO_MUTATE_MAX_TOKENS` | `512` (mutation rewrite cap) |
| `maspo/integration.py: MASPOSettings` | beam 2, 2 offspring, minibatch 10, depth ≤ 9, 3 rounds per turn, 12 threads, 4 judge calls in flight |
| `maspob/regime.py: MASPOBSettings` | 20 variants per role, generation temperature 0.5, MiniLM embeddings, seed 42 (+1000 per optimizer seed), minibatch 5, ≤ 5 warm-up pulls |
| `tavo/settings.py: TAVOSettings` | train batch 6, ≤ 5 outer rounds, validation batch ≥ 3, adoption threshold 0.01, 2 attempts per round, patience 2, temperature 0.5 |
| `TAVO_CREDIT` | `1`; `0`/`false`/`no`/`off` = no trajectory credit (ablation) |

### Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider optimizers --ignore=optimizers/bridge -q
```

The tests use fakes (`optimizers/conftest.py`, `optimizers/protocol/tests/fakes.py`) and need no endpoint; MASPOB's GNN tests skip without `torch_geometric`. The `methods` golden cells ([`tests/golden/methods.py`](../../tests/golden/methods.py)) pin each method's end-to-end trace.

---

## Optimizer interface

A method is a package `optimizers/<method>/` whose `integration.py` exposes the optimizer class, registered in `methods/__init__.py` (`METHODS`).

```python
class MyOptimizer:
    def __init__(self, seed_bundle: PromptBundle, *, run_dir=None, reflection_lm=None): ...
    def optimize(self, cell: CellSpec, runner, budget: BudgetLedger,
                 training: list[dict], validation: list[dict]) -> OptimizerResult: ...
```

- **Constructor** (`methods.build_optimizer`): the seed bundle as `seed_bundle` or `initial_bundle`, whichever the signature declares; if declared, `run_dir` (scratch directory), `reflection_lm` (`dspy_bridge.build_reflection_lm()`) and `task_lm` (`dspy_bridge.build_task_lm()`).
- **Rows**: JSON-safe dicts with an `id`, `task_instance` and gold labels; pass them to the runner unchanged. Test rows never reach the optimizer.
- **Runner** (`ProtocolRunner`): `run(example, bundle, request_seed) -> RunRecord`, `run_batch(examples, bundle, seeds) -> BatchExecution`, plus `budget`, `cell`, `phase`, `seed_bundle`/`initial_bundle`, `required_roles`, `validate_bundle`, `artifact_store`, `artifact_directory`, `supports_concurrent = False`. A record carries `score`, `status`, `final_output`, `messages`, `usage` and the metric feedback in `metadata["scorer_metadata"]["feedback"]`.
- **Return**: a `schema.OptimizerResult` (`method`, `cell_id`, seed and incumbent bundles, `budget_snapshot == budget.snapshot()`, a `schema.StopReason`), published as `SEARCH_LAYOUT` (incumbent as `selected_bundle`) or `JOURNAL_LAYOUT` (`RunnerSession` methods and identity; `incumbent_bundle`, via `journal.LifecycleJournal`).
- **Errors**: all in `errors.py`; an infrastructure failure (`PreObservationInfrastructureFailure`, `NativeInfrastructureExhausted`, `OptimizerInfrastructureFailure`) keeps the seed as `infrastructure_invalid`.
- **Masked execution**: a bundle with `metadata["optimizer_control"]` runs through the hook registered by `runner.register_execution_hook("optimizer_control", hook)` (`build_adapter(runtime, request, control)`, `verify(runtime, request, control, output)`), else it is rejected.
