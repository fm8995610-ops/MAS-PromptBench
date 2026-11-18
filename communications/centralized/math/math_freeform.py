"""Communications pair: centralized MATH with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="math", fmt="freeform")
