"""Iterative Refinement LiveCodeBench runner (LangGraph): a generator, two read-only critics and a verifier refine one candidate for at most three rounds.

Runs topologies.workflows.lcb with TOPOLOGY="iterative_refinement" (see core.variant); the module owns its hooks.
"""

from core import variant

variant.load(globals(), "topologies.workflows.lcb", TOPOLOGY="iterative_refinement")
