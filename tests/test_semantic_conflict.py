"""Semantic conflict and source validation tests."""

import pytest

from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.conflict.semantic import (
    is_cross_relation_duplicate,
    is_object_redundant,
    should_invalidate,
)
from mnemo.app.services.extraction.validators import (
    dedupe_batch_facts,
    is_technology_object,
    validate_fact_against_source,
)


class _Edge:
    def __init__(self, relation: str, obj: str, id: str = "e1"):
        self.id = id
        self.relation = relation
        self.object = obj
        self.fact_string = f"user {relation.lower()} {obj}"


def _fact(relation, obj, **kw):
    return TripletFact(
        subject="user",
        relation=relation,
        object=obj,
        fact_string=f"user {relation.lower()} {obj}",
        confidence=0.95,
        **kw,
    )


def test_role_slot_invalidates_old_role():
    new = _fact("IS", "ai engineer")
    old = _Edge("HAS_ROLE", "backend engineer", "e1")
    assert should_invalidate(new, old) is True


def test_cross_relation_duplicate_same_role():
    new = _fact("IS", "backend engineer")
    old = _Edge("HAS_ROLE", "backend engineer", "e1")
    assert is_cross_relation_duplicate(new, old) is True


def test_employment_object_redundant():
    new = _fact("SWITCHED_FROM", "Slice")
    old = _Edge("WORKED_AT", "Slice", "e1")
    assert is_object_redundant(new, old) is True
    assert should_invalidate(new, old) is True


def test_lives_in_supersedes_born_in_same_city():
    new = _fact("LIVES_IN", "Delhi")
    old = _Edge("BORN_IN", "Delhi", "e1")
    assert should_invalidate(new, old) is True


def test_reject_born_in_from_living_language():
    text = "I have been living in Delhi for two years."
    fact = _fact("BORN_IN", "Delhi")
    assert validate_fact_against_source(text, fact) is False
    lives = _fact("LIVES_IN", "Delhi")
    assert validate_fact_against_source(text, lives) is True


def test_reject_transition_relations():
    text = "I switched from Slice to Deepmind and use Python."
    assert validate_fact_against_source(text, _fact("SWITCHED_FROM", "Slice")) is False
    assert validate_fact_against_source(text, _fact("SWITCHED_TO", "Deepmind")) is False
    assert validate_fact_against_source(text, _fact("USES", "Python")) is True


def test_works_at_invalidates_past_employment():
    new = _fact("WORKS_AT", "Deepmind")
    old = _Edge("WORKED_AT", "Slice", "e1")
    assert should_invalidate(new, old) is True


def test_stack_profile_uses_technology_only():
    assert is_technology_object("Python") is True
    assert is_technology_object("Deepmind") is False


def test_dedupe_drops_past_employment_when_current_present():
    facts = [
        _fact("WORKS_AT", "Deepmind"),
        _fact("WORKED_AT", "Slice"),
        _fact("SWITCHED_FROM", "Slice"),
    ]
    deduped = dedupe_batch_facts(facts)
    assert {f.relation for f in deduped} == {"WORKS_AT"}


def test_dedupe_batch_keeps_current_role_by_source_position():
    text = (
        "I worked at Slice as a backend engineer but now I am working at Deepmind "
        "as ai engineer and mainly use Python."
    )
    facts = [
        _fact("IS", "backend engineer"),
        _fact("HAS_ROLE", "ai engineer"),
    ]
    deduped = dedupe_batch_facts(facts, source_text=text)
    assert len(deduped) == 1
    assert deduped[0].object == "ai engineer"


def test_dedupe_batch_keeps_last_role():
    facts = [
        _fact("IS", "backend engineer"),
        _fact("IS", "ai engineer"),
    ]
    deduped = dedupe_batch_facts(facts)
    assert len(deduped) == 1
    assert deduped[0].object == "ai engineer"
