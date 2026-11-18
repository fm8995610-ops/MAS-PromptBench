"""Iterative Refinement API-Bank runner (LangGraph): a generator, two read-only critics and a verifier refine one candidate for at most three rounds.

Runs topologies.workflows.apibank with TOPOLOGY="iterative_refinement" (see core.variant); the module owns its hooks.
"""

from core import variant

variant.load(globals(), "topologies.workflows.apibank", TOPOLOGY="iterative_refinement")
