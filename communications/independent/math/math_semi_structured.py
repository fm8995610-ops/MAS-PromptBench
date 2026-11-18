"""Communications pair: independent MATH with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="math", fmt="semi_structured")
