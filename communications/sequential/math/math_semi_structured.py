"""Communications pair: sequential MATH with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="sequential", dataset="math", fmt="semi_structured")
