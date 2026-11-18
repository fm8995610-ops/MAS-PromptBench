# Team specs

This folder declares what **team size `r`** means for each multi-agent topology, one `<dataset>.yaml` per dataset (top-level key = dataset), and the one four-agent team of each extension topology. API-Bank and ToolHop declare only the extension topologies, whose tools come with each instance: for the other topologies they fix their roles in code and vote over r replicas (`teamsizes/apibank_common.py`, `toolhop_common.py`).

## Layout

```
configs/teams/
└── <dataset>.yaml      # apps, bfcl, gpqa, hotpotqa, lcb, math, swe; apibank, toolhop (extension topologies only)
```

| Topology | Keys |
|---|---|
| `independent` | `role` (r replicas), `recursion_limit` |
| `decentralized` | `role` (r peers), `n_rounds`, `recursion_limit` |
| `sequential` | `recursion_limit`, `tasks` (each stage's user-message template, `str.format`), `sizes`: r → the r ordered `{role, tools}` stages |
| `centralized` | `recursion_limit`, `manager_tools`, `worker_tools` (per-worker override: `tools_by_worker`), `prompt_suffixes` (optional, appended to a worker's prompt), `delegates` (each `delegate_to_<worker>` tool description, indentation included), `sizes`: r → `manager` (prompt key), `max_turns`, `workers` (r − 1) |
| `fan_out_fan_in` | `recursion_limit`, `tools`: `coordinator` and `worker` → tool names (3 workers share the `worker` role) |
| `iterative_refinement` | `recursion_limit`, `max_rounds` (3), `tools`: `generator`, `critic_1`, `critic_2`, `verifier` → tool names |

---

## How specs are used

- **Loaded** by `core/teams.py`: `teams.spec` returns one `TeamSpec` per `(topology, dataset, r)`, r ∈ {2, 4, 8, 10}; the extension topologies (`FIXED_TEAMS`) exist only at r = 4.
- **r = 4** is the team of the `topologies/` runners (all but CrewAI); `teamsizes/` runners use their own r.
- **`recursion_limit`** bounds each agent's ReAct loop; omit it when agents run no ReAct loop (API-Bank, ToolHop).
- **Read-only roles** — on SWE-bench the `iterative_refinement` critics and verifier list only `file_read`, `list_dir` and `search_repo`; the runner refuses a read-only role with an editing tool.
- **Verbatim** — text is sent to the model as written.
