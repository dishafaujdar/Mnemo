"""Tests for object normalization, temporal guard, and pre-insert dedup."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.conflict.pre_insert_dedup import should_skip_as_duplicate
from mnemo.app.services.conflict.temporal import safe_invalid_at
from mnemo.app.services.ontology.canonical import normalize_object, object_word_count


def test_normalize_object_max_three_words():
    raw = "Deepmind as ai engineer and mainly use Python and Cursor"
    assert normalize_object(raw) == "Deepmind as ai"


def test_object_word_count():
    assert object_word_count("Deepmind as ai engineer") == 4


def test_safe_invalid_at_never_before_valid_at():
    valid_at = datetime(2026, 7, 21, 7, 30, 39)
    earlier = datetime(2026, 7, 21, 7, 22, 4)
    result = safe_invalid_at(valid_at, earlier)
    assert result > valid_at
    assert result == valid_at + timedelta(milliseconds=1)


def test_safe_invalid_at_uses_new_fact_when_later():
    valid_at = datetime(2026, 7, 21, 7, 22, 4)
    later = datetime(2026, 7, 21, 7, 30, 39)
    assert safe_invalid_at(valid_at, later) == later


@pytest.mark.asyncio
async def test_pre_insert_dedup_exact_object_match():
    edge = MagicMock()
    edge.object = "Deepmind"
    edge.relation = "WORKS_AT"
    edge.fact_string = "user works at Deepmind"
    edge.invalid_at = None

    db = AsyncMock()
    db.execute = AsyncMock(
        return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=lambda: [edge])))
    )

    fact = TripletFact(
        subject="user",
        relation="WORKS_AT",
        object="Deepmind",
        fact_string="user works at Deepmind",
        confidence=0.95,
    )
    assert await should_skip_as_duplicate(db, "u_01", fact) is True
