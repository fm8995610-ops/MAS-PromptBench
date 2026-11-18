"""Communications pair: independent MATH with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="math", fmt="structured_soft")
