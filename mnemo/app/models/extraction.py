"""Pydantic schemas for fact extraction (triplets)."""

from __future__ import annotations

from pydantic import BaseModel

# Review status values carried on a fact through validation/storage.
REVIEW_CONFIRMED = "confirmed"  # canonical relation, exact ontology match
REVIEW_FUZZY = "fuzzy"  # matched via fuzzy alias, auto-learned
REVIEW_PENDING = "pending"  # unknown relation, stored but flagged for audit
REVIEW_UNKNOWN = "unknown"  # could not map to a canonical relation; skip storage
REVIEW_REJECTED = "rejected"  # discarded (kept only for debugging/telemetry)


class TripletFact(BaseModel):
    """A single extracted fact: subject --[relation]--> object.

    Core fields (subject/relation/object/fact_string/confidence) are the stable
    contract used by the conflict resolver and storage layer. The remaining
    fields carry provenance from the two-stage GLiNER2 + LLM pipeline and are
    optional so older callers keep working.
    """

    subject: str
    relation: str
    object: str
    fact_string: str
    confidence: float = 1.0

    # Grounding + temporal pipeline fields
    source_span: str = ""
    temporal_status: str = "unspecified"  # current | past | future | unspecified

    # Provenance / soft-ontology metadata (optional, defaulted for back-compat).
    relation_raw: str = ""  # what the extractor originally emitted
    relation_match_score: float = 1.0  # fuzzy ontology match score (0..1)
    review_status: str = REVIEW_CONFIRMED
    reasoning: str = ""  # LLM justification grounded in raw text
    temporal_hint: str | None = None  # e.g. "last month", "since 2020"
    source: str = ""  # "gliner" | "llm"
    retraction_signal: bool = False  # historical fact: store then close immediately


class ExtractionResult(BaseModel):
    """Output of the extraction pipeline including episode-level retraction hints."""

    facts: list[TripletFact] = []
    retract_others_in_category: str | None = None  # e.g. "USES" = exhaustive tool list
