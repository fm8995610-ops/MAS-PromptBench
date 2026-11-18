"""Iterative Refinement GPQA runner (LangGraph): a generator, two read-only critics and a verifier refine one candidate for at most three rounds.

Runs topologies.workflows.gpqa with TOPOLOGY="iterative_refinement" (see core.variant); the module owns its hooks.
"""

from core import variant

variant.load(globals(), "topologies.workflows.gpqa", TOPOLOGY="iterative_refinement")
