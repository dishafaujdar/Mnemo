"""Apply temporal labels from Step 1 to extracted facts."""

from __future__ import annotations

from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.extraction.temporal_classifier import (
    ClassifiedClause,
    temporal_status_for_span,
)


def apply_temporal_metadata(
    facts: list[TripletFact],
    classified: list[ClassifiedClause],
) -> list[TripletFact]:
    """Set temporal_status and retraction_signal from clause classification."""
    updated: list[TripletFact] = []
    for fact in facts:
        status = fact.temporal_status
        if not status or status == "unspecified":
            status = temporal_status_for_span(fact.source_span, classified)
        if status == "unspecified":
            status = "current"
        retraction = status == "past"
        updated.append(
            fact.model_copy(
                update={
                    "temporal_status": status,
                    "retraction_signal": retraction,
                }
            )
        )
    return updated
