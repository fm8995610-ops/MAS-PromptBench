"""Fan-out/Fan-in GPQA runner (LangGraph): a coordinator decomposes once, three workers sharing one prompt solve the subtasks in parallel, the coordinator synthesizes.

Runs topologies.workflows.gpqa with TOPOLOGY="fan_out_fan_in" (see core.variant); the module owns its hooks.
"""

from core import variant

variant.load(globals(), "topologies.workflows.gpqa", TOPOLOGY="fan_out_fan_in")
