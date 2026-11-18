"""Fan-out/Fan-in SWE-bench Verified runner (LangGraph): a coordinator decomposes once, three workers sharing one prompt solve the subtasks in parallel, the coordinator synthesizes.

Runs topologies.workflows.swe with TOPOLOGY="fan_out_fan_in" (see core.variant); the module owns its hooks.
"""

from core import variant

variant.load(globals(), "topologies.workflows.swe", TOPOLOGY="fan_out_fan_in")
