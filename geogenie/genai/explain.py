"""Result rationale generation (template or LLM)."""

from __future__ import annotations

from typing import List, Optional, Sequence

from geogenie.core.types import POI, SearchResult, StructuredQuery


def explain_results(
    query: StructuredQuery,
    results: Sequence[SearchResult],
    excluded_sample: Optional[Sequence[POI]] = None,
) -> List[str]:
    """Return one explanation string per result."""
    explanations = []
    for r in results:
        parts = []
        if r.poi.accessible:
            parts.append("step-free / accessible")
        if r.poi.noise_level:
            parts.append(f"noise: {r.poi.noise_level}")
        if r.poi.category:
            parts.append(r.poi.category)
        mins = query.minutes
        if mins is not None:
            parts.append(f"within ~{mins:g} min walk ({r.distance_m:.0f} m)")
        else:
            parts.append(f"{r.distance_m:.0f} m away")
        reason = "; ".join(parts) if parts else "matches your search"
        explanations.append(f"Selected: {reason}.")
    return explanations
