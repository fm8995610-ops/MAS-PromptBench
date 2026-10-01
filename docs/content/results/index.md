---
title: Results Explorer
description: Every GEPA result from the MAS-PromptBench paper by task, topology, framework, communication protocol and team size.
---

# Results Explorer

Every GEPA result the paper reports, in one place. Each cell is one configuration scored with its seed prompts and again with the optimized prompts on the same examples. Hover a cell for its full name.
{ .lede }

--8<-- "results/headline.html"

## By task and topology

The **topology study** runs GEPA on the five topologies. The **framework study** runs the same five on popular multi-agent frameworks, one framework per topology. Single and Independent use LangGraph in both, so those two columns match.

=== "Topology study"

    --8<-- "results/grid-topology-study.html"

=== "Framework study"

    --8<-- "results/grid-framework-study.html"

Read a column top to bottom to see how one topology behaves across tasks, and a row left to right to see one task across topologies. Large blue and orange cells sit side by side in most rows: the same optimizer, on the same task, helps one coordination structure and hurts another.

## Averages

<div class="mini-grid" markdown>

<div markdown>

### Task domain

--8<-- "results/summary-domains.html"

</div>

<div markdown>

### Topology

--8<-- "results/summary-topologies.html"

</div>

<div markdown>

### Communication protocol

--8<-- "results/summary-protocols.html"

</div>

<div markdown>

### Team size

--8<-- "results/summary-team-sizes.html"

</div>

</div>

## Communication protocols

HotpotQA and LiveCodeBench, each run with the four multi-agent topologies under three message formats. See [Communication Protocols](../mas/communication-protocols.md).

--8<-- "results/protocols.html"

## Team sizes

The same two tasks with 2, 4, 8 and 10 agents. See [Team Sizes](../mas/team-sizes.md).

--8<-- "results/team-sizes.html"

## How to read these numbers

- **Δ** is the optimized score minus the baseline score, in percentage points. A run's `meta.json` stores the same quantity as a fraction (`delta`); multiply by 100 to compare. See [Read the Results](../evaluation/results.md).
- **Scores** are the task's own metric: accuracy, exact match, pass@1 or resolve rate. Each task page names its metric.
- **Color** follows the paper's convention: blue for a gain, orange for a regression, gray for no change. Darker means larger, in steps at 2, 5 and 10 points.
- **The protocol and team-size cells** report Δ only, as the paper's figures do.

!!! note "Source"
    All numbers are the GEPA results reported in the paper and on the [project page](https://fm8995610-ops.github.io/MAS-PromptBench/). To reproduce a cell, follow the [Quick Start](../getting-started/quick-start.md) with that cell's task and topology.
