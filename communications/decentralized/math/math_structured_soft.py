"""Communications pair: decentralized MATH with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="math", fmt="structured_soft")
