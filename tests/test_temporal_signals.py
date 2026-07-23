"""Tests for past-tense retraction and exhaustive USES detection."""

from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.extraction.temporal_signals import (
    apply_temporal_signals,
    infer_retraction_signal,
    infer_retract_others_in_category,
)


def _fact(relation: str, obj: str, **kw) -> TripletFact:
    return TripletFact(
        subject="user",
        relation=relation,
        object=obj,
        fact_string=f"user {relation.lower()} {obj}",
        confidence=0.9,
        **kw,
    )


def test_past_tense_sets_retraction_signal():
    text = "I used to work at Slice but now I work at Deepmind"
    fact = _fact("WORKED_AT", "Slice")
    assert infer_retraction_signal(text, fact) is True


def test_worked_at_always_retraction_signal():
    text = "I work at Deepmind"
    fact = _fact("WORKED_AT", "Slice")
    assert infer_retraction_signal(text, fact) is True


def test_exhaustive_uses_detected():
    text = "I mainly use Python and Cursor for everything"
    facts = [_fact("USES", "Python"), _fact("USES", "Cursor")]
    assert infer_retract_others_in_category(text, facts) == "USES"


def test_apply_temporal_signals_annotates_facts():
    text = "I used to work at Slice and mainly use Python"
    facts = [_fact("WORKED_AT", "Slice"), _fact("USES", "Python")]
    updated, category = apply_temporal_signals(text, facts)
    assert updated[0].retraction_signal is True
    assert category == "USES"
