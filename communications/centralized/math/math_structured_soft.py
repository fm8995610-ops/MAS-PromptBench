"""Communications pair: centralized MATH with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="math", fmt="structured_soft")
