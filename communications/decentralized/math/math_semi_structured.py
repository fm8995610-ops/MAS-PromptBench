"""Communications pair: decentralized MATH with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="math", fmt="semi_structured")
