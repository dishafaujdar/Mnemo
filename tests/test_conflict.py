"""Conflict resolver tests: duplicate and contradiction detection."""

import pytest

from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.conflict.resolver import is_contradiction, is_duplicate


class _MockEdge:
    def __init__(self, obj: str):
        self.object = obj


def test_is_duplicate_true():
    fact = TripletFact(subject="user", relation="PREFERS", object="Python", fact_string="User prefers Python", confidence=0.9)
    existing = [_MockEdge("Python")]
    assert is_duplicate(fact, existing) is True


def test_is_duplicate_false():
    fact = TripletFact(subject="user", relation="PREFERS", object="Go", fact_string="User prefers Go", confidence=0.9)
    existing = [_MockEdge("Python")]
    assert is_duplicate(fact, existing) is False


def test_is_contradiction_singular():
    fact = TripletFact(subject="user", relation="SWITCHED_TO", object="Go", fact_string="User switched to Go", confidence=0.9)
    existing = [_MockEdge("Python")]
    assert is_contradiction(fact, existing) is True


def test_is_contradiction_multi_value_relation():
    fact = TripletFact(subject="user", relation="USES", object="Rust", fact_string="User uses Rust", confidence=0.9)
    existing = [_MockEdge("Python")]
    assert is_contradiction(fact, existing) is False


def test_is_contradiction_same_object():
    fact = TripletFact(subject="user", relation="IS", object="developer", fact_string="User is developer", confidence=0.9)
    existing = [_MockEdge("developer")]
    assert is_contradiction(fact, existing) is False
