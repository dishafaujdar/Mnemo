"""Past-tense retraction and exhaustive-category detection from source text."""

from __future__ import annotations

import re

from mnemo.app.models.extraction import TripletFact

_PAST_TENSE_NEAR_OBJECT = re.compile(
    r"\b(?:"
    r"used to work(?:\s+at|\s+for)?|"
    r"previously worked(?:\s+at|\s+for)?|"
    r"was at|"
    r"worked at|"
    r"worked for|"
    r"had worked(?:\s+at|\s+for)?"
    r")\b",
    re.IGNORECASE,
)

_EXHAUSTIVE_USES = re.compile(
    r"\b(?:"
    r"mainly use|mostly use|primarily use|"
    r"only use|just use|"
    r"work with|working with|"
    r"I use"
    r")\b",
    re.IGNORECASE,
)


def _object_head(obj: str) -> str:
    return obj.strip().lower().split()[0] if obj.strip() else ""


def infer_retraction_signal(source_text: str, fact: TripletFact) -> bool:
    """True when the source describes this fact in past tense (historical, not current)."""
    if fact.retraction_signal:
        return True
    if fact.temporal_status == "past":
        return True
    if fact.relation.upper() == "WORKED_AT":
        return True
    text = source_text.lower()
    head = _object_head(fact.object)
    if not head or head not in text:
        return False
    # Past-tense employment/location phrasing near the entity.
    if fact.relation.upper() in {"WORKS_AT", "WORKED_AT", "LIVES_IN", "LIVED_IN"}:
        idx = text.find(head)
        window = text[max(0, idx - 60) : idx + len(head) + 20]
        if _PAST_TENSE_NEAR_OBJECT.search(window):
            return True
    return False


def infer_retract_others_in_category(
    source_text: str,
    facts: list[TripletFact],
    llm_hint: str | None = None,
) -> str | None:
    """Detect exhaustive tool lists that imply retracting other USES facts."""
    if llm_hint:
        return llm_hint.upper()
    if not _EXHAUSTIVE_USES.search(source_text):
        return None
    uses = [f for f in facts if f.relation.upper() == "USES" and not f.retraction_signal]
    return "USES" if uses else None


def apply_temporal_signals(
    source_text: str,
    facts: list[TripletFact],
    *,
    retract_others_in_category: str | None = None,
) -> tuple[list[TripletFact], str | None]:
    """Annotate facts with retraction_signal; return category retraction hint."""
    updated: list[TripletFact] = []
    for fact in facts:
        signal = infer_retraction_signal(source_text, fact)
        updated.append(fact.model_copy(update={"retraction_signal": signal}))
    category = infer_retract_others_in_category(
        source_text, updated, llm_hint=retract_others_in_category
    )
    return updated, category
