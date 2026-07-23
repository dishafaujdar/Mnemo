"""Fact validation tiers.

Two independent signals gate a fact before storage:

1. **Ontology match tier** (how well the relation maps to the ontology) ->
   sets ``review_status`` (confirmed / fuzzy / pending). Unknown relations are
   kept (flagged pending), never discarded.
2. **Judge score** (LLM self-assessed support against the raw user text, carried
   on ``confidence``) -> decides store / review / discard.
"""

from __future__ import annotations

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import (
    REVIEW_CONFIRMED,
    REVIEW_FUZZY,
    REVIEW_PENDING,
    REVIEW_REJECTED,
    REVIEW_UNKNOWN,
    TripletFact,
)
from mnemo.app.services.ontology.manager import (
    TIER_CONFIRMED,
    TIER_FUZZY,
    TIER_REJECT,
    TIER_UNKNOWN,
    get_ontology,
)

ACTION_STORE = "store"
ACTION_REVIEW = "review"
ACTION_DISCARD = "discard"

_TIER_TO_STATUS = {
    TIER_CONFIRMED: REVIEW_CONFIRMED,
    TIER_FUZZY: REVIEW_FUZZY,
    TIER_UNKNOWN: REVIEW_UNKNOWN,
    TIER_REJECT: REVIEW_REJECTED,
}


def review_status_for_tier(tier: str) -> str:
    """Map an ontology match tier to a stored review status."""
    return _TIER_TO_STATUS.get(tier, REVIEW_PENDING)


def judge_action(judge_score: float) -> str:
    """Route a fact by its judge score (grounding against raw user text)."""
    if judge_score >= settings.judge_store_threshold:
        return ACTION_STORE
    if judge_score >= settings.judge_review_threshold:
        return ACTION_REVIEW
    return ACTION_DISCARD


def apply_judge(facts: list[TripletFact]) -> list[TripletFact]:
    """Filter LLM-extracted facts by judge score and record unknown relations.

    - ``store``  -> kept as-is.
    - ``review`` -> kept but forced to ``pending`` review status.
    - ``discard``-> dropped.
    """
    ontology = get_ontology()
    kept: list[TripletFact] = []
    for fact in facts:
        if not fact.relation or not fact.object.strip():
            continue
        action = judge_action(fact.confidence)
        if action == ACTION_DISCARD:
            continue
        if action == ACTION_REVIEW and fact.review_status == REVIEW_CONFIRMED:
            fact.review_status = REVIEW_PENDING
        if fact.review_status == REVIEW_PENDING:
            ontology.record_unknown(fact.relation, fact.confidence)
        kept.append(fact)
    return kept
