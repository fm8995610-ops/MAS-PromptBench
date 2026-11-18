"""Fan-out/Fan-in MATH runner (LangGraph): a coordinator decomposes once, three workers sharing one prompt solve the subtasks in parallel, the coordinator synthesizes.

Runs topologies.workflows.math with TOPOLOGY="fan_out_fan_in" (see core.variant); the module owns its hooks.
"""

from core import variant

variant.load(globals(), "topologies.workflows.math", TOPOLOGY="fan_out_fan_in")
