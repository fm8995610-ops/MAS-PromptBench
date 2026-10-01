# Ask DeepWiki

DeepWiki is an AI-generated wiki of the MAS-PromptBench repository that you can ask questions in plain language. Use it to find where something lives in the code; use these docs and the code itself as the source of truth.
{ .lede }

<div class="deepwiki-panel" markdown>

[Open MAS-PromptBench on DeepWiki](https://deepwiki.com/fm8995610-ops/MAS-PromptBench){ .btn .btn-deepwiki }

`deepwiki.com/fm8995610-ops/MAS-PromptBench`

</div>

## What it is good for

DeepWiki, from Cognition, reads the repository and writes an overview with linked pages and diagrams. Its chat answers questions with references to files and lines. It works well for:

- **Finding code.** "Where is the HotpotQA exact-match scorer?" or "Which file builds the centralized AutoGen group chat?"
- **Tracing a call path.** "What happens between `run_gepa_dataset` and the runner's solve function?"
- **Orientation in a new folder.** "Summarize what `communications/communication_formats.py` does."

These docs cover how to run the benchmark and what the results mean. DeepWiki covers the code in more depth than any page here.

!!! warning "Check before you rely on an answer"
    DeepWiki's pages and answers are generated and can be wrong or out of date. Before you use a flag, path or default in an experiment, confirm it in the code or in the [Command-Line Flags](cli.md) and [Environment Variables](environment.md) references.

## Questions to try

```text
Which environment variables does the GEPA pilot read, and what are their defaults?
How does the independent topology aggregate the four replicas' answers?
Where are the frozen eval IDs excluded from the optimizer's train split?
What does the structured_soft communication format require of each message?
How do I add a new dataset adapter for MIPRO?
```

## Use it from a coding agent

DeepWiki also runs a free MCP server, so an assistant such as Claude Code or Cursor can query the wiki while it works in your checkout. No login is needed for public repositories.

```json title="MCP client configuration"
{
  "mcpServers": {
    "deepwiki": {
      "url": "https://mcp.deepwiki.com/mcp"
    }
  }
}
```

It exposes three tools: `read_wiki_structure` (list the wiki's topics), `read_wiki_contents` (read them) and `ask_question` (ask about a repository, here `fm8995610-ops/MAS-PromptBench`).

## For maintainers

### Index the repository

DeepWiki builds wikis for public repositories on request. Open [deepwiki.com](https://deepwiki.com), submit `https://github.com/fm8995610-ops/MAS-PromptBench`, and wait for the first build. After that, the link above shows the wiki.

### Keep it fresh with the badge

Add the badge to the top of `README.md`. Besides linking readers to the wiki, a repository that carries the badge gets its wiki refreshed automatically as the code changes.

```markdown title="README.md"
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fm8995610-ops/MAS-PromptBench)
```

### Steer what it writes

A `.devin/wiki.json` file in the repository root tells DeepWiki which pages to write and what to stress. It holds `repo_notes` (context, up to 10,000 characters each) and `pages` (each with a unique `title`, a `purpose`, an optional `parent` and optional `page_notes`), with at most 30 pages. A version that mirrors these docs and states the gotchas readers hit most often:

```json title=".devin/wiki.json (excerpt)"
{
  "repo_notes": [
    {
      "content": "Run every runner as a module from the repo root (python -m topologies.<...>); the file-path form fails with ModuleNotFoundError. Optimizers read GEPA_TASK_ENDPOINTS / GEPA_REFL_ENDPOINT, not VLLM_BASE_URL."
    }
  ],
  "pages": [
    { "title": "Overview", "purpose": "What the benchmark measures: the prompt-optimization gain across tasks, topologies, protocols and team sizes." },
    { "title": "Workflow Topologies", "purpose": "The five topologies and their LangGraph, CrewAI, AutoGen and OpenAI SDK implementations." },
    { "title": "Sequential", "parent": "Workflow Topologies", "purpose": "The four-stage pipeline and its role prompts." }
  ]
}
```

The full file ships with these docs as `.devin/wiki.json`.
