# Known Issues

Places where the repository's code, scripts or READMEs disagree, with the workaround these docs use and the file to change. Once an issue is fixed, delete its row and the matching note on the page it links to.
{ .lede }

<div class="facts" markdown>
<div><span>Breaks a run</span>7</div>
<div><span>Surprising default</span>4</div>
<div><span>Docs mismatch</span>4</div>
</div>

## Breaks a run

| Issue | Workaround | Fix in |
| --- | --- | --- |
| Most runner files import `topologies` before adding the repo root to `sys.path`, so `python topologies/.../x.py` fails with `ModuleNotFoundError`. | Run modules: `python -m topologies.<...>`, or prefix `PYTHONPATH=.`. See [Installation](../getting-started/installation.md#check-the-install). | the import block at the top of each runner under `topologies/` and `teamsizes/` |
| `environment.yml` pins `dspy>=2.6,<3`; `dspy.teleprompt.GEPA` only exists in DSPy 3. | `pip install -U "dspy[optuna]>=3"` in the environment. | `environment.yml` |
| `run_topologies.sh` and `run_teamsizes.sh` pass `--batch` (and `--out`) to the BFCL and SWE-bench runners, which reject both. | Run those cells directly with `--limit` and `--out-dir`. See [BFCL](../tasks/bfcl.md#run-it). | `scripts/run_topologies.sh`, `scripts/run_teamsizes.sh` |
| `run_teamsizes.sh` passes no output path, so GPQA, HotpotQA, MATH, LiveCodeBench and APPS team-size runs save nothing. | Run those cells directly with `--out`. See [Team Sizes](../mas/team-sizes.md#run-a-cell). | `scripts/run_teamsizes.sh` |
| SWE-bench communication-protocol cells call the runner as `run_one(instance, None)`, missing its output directory, so every row is written as an error. | None; skip SWE-bench in the protocol study. | `communications/communication_formats.py` (`run_batch`) |
| `optimizers/gepa/scripts/smoke_new_adapters.py` imports a `travel` dataset module that doesn't exist. | Don't run it; smoke-test a new pair with a small `--train-size`. | `optimizers/gepa/scripts/smoke_new_adapters.py` |
| The SWE-bench optimizer adapter looks for checkouts in `SWE_WORK_ROOT`, then under `~/swe_work*`, and silently falls back to a tiny placeholder repository. | Set `SWE_WORK_ROOT` to your checkouts and `REQUIRE_REAL_SWE_WORKDIR=1` so a missing checkout fails loudly. See [SWE-bench Verified](../tasks/swe-bench.md). | `optimizers/*/real_runner_*/adapters/module_swe.py` |

## Surprising defaults

| Issue | Workaround | Fix in |
| --- | --- | --- |
| The runners default `VLLM_BASE_URL` to two different local ports (`http://localhost:8000/v1`, `http://localhost:8001/v1`), and the optimizers default to a long list of local ports. | Always export the endpoint variables. See [Connect a Model](../getting-started/connect-a-model.md). | the runner headers; `optimizers/*/real_runner_*/lm.py` |
| The optimizers ignore `VLLM_BASE_URL` for task calls, and `run_gepa.sh` never sets `GEPA_TASK_ENDPOINTS`. | Export `GEPA_TASK_ENDPOINTS` / `MIPRO_TASK_ENDPOINTS`. | `optimizers/gepa/run_gepa.sh`, the optimizer READMEs |
| The GEPA and MIPRO pilots default to `--n-agents 2 --n-rounds 1`, while the runners use 4 agents and 2 debate rounds; `run_mipro.sh` passes neither flag. | Pass `--n-agents 4 --n-rounds 2` for Independent and Decentralized cells. See [Quick Start](../getting-started/quick-start.md#5-try-a-multi-agent-cell). | `run_gepa_dataset.py`, `run_mipro_dataset.py`, `run_mipro.sh` |
| The sweep scripts advertise a `<DATASET>_LIMIT` override that they never read; limits are hard-coded. | Edit the `LIMIT` table in the script. | `scripts/run_*.sh` |

## Docs and code disagree

| README says | Code does | Fix in |
| --- | --- | --- |
| A `.env` file with provider blocks ships with the repo. | No `.env` or `.env.example` is committed. | `README.md`, or commit `.env.example` |
| Every runner has a smoke demo and writes to `results/<dataset>/`. | BFCL, SWE-bench, ToolHop and API-Bank run a batch instead of a demo; GPQA, HotpotQA, MATH, LiveCodeBench and APPS save only with `--out`; other defaults vary by runner. | `README.md`, `topologies/README.md` |
| Eval-ID exclusion is switched on with `*_EXCLUDE_REAL_EVAL_IDS=1`. | It is on unless set to `0`, `false`, `no` or `off`. | `optimizers/*/README.md` |
| Runners load the frozen eval IDs. | Only the API-Bank runners read their manifest; pass the IDs with `--only` elsewhere. See [Evaluation Protocol](../evaluation/protocol.md). | `benchmarks/README.md` |
