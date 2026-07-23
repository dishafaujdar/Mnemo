"""Tests for the restructured extraction pipeline (Steps 1–5)."""

import pytest

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import REVIEW_CONFIRMED, TripletFact
from mnemo.app.services.extraction import gliner_extractor, pipeline, structured_extractor
from mnemo.app.services.extraction.grounding import ensure_source_spans, validate_grounding
from mnemo.app.services.extraction.relation_guards import validate_lives_in
from mnemo.app.services.extraction.service import gate_facts
from mnemo.app.services.extraction.temporal_apply import apply_temporal_metadata
from mnemo.app.services.extraction.temporal_classifier import (
    ClassifiedClause,
    classify_clauses,
    split_clauses,
    temporal_status_for_span,
)
from mnemo.app.services.extraction.temporal_signals import apply_temporal_signals

TEST_INPUT = (
    "I used to play football but switched to basketball two years ago. "
    "I am learning guitar since last month. "
    "I live with two cats named Luna and Milo."
)


def _fact(**kw) -> TripletFact:
    defaults = dict(
        subject="user",
        relation="USES",
        object="Python",
        fact_string="user uses Python",
        confidence=0.95,
        review_status=REVIEW_CONFIRMED,
        source_span="I use Python",
    )
    defaults.update(kw)
    return TripletFact(**defaults)


# --- Step 1: temporal classification ----------------------------------------
def test_split_clauses_splits_on_but():
    parts = split_clauses(
        "I used to play football but switched to basketball two years ago."
    )
    assert len(parts) == 2
    assert "used to play football" in parts[0].lower()
    assert "switched to basketball" in parts[1].lower()


@pytest.mark.asyncio
async def test_classify_clauses_heuristic_without_api(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "")

    async def _heuristic_only(sentence: str):
        from mnemo.app.services.extraction import temporal_classifier as tc

        return tc._heuristic_label(sentence)

    monkeypatch.setattr(
        "mnemo.app.services.extraction.temporal_classifier.classify_temporal_status",
        _heuristic_only,
    )
    classified = await classify_clauses(TEST_INPUT)
    by_text = {c.text.lower(): c.temporal_status for c in classified}
    assert any("used to play football" in t and s == "past" for t, s in by_text.items())
    assert any("switched to basketball" in t and s == "current" for t, s in by_text.items())
    assert any("learning guitar" in t and s == "current" for t, s in by_text.items())
    assert any("live with two cats" in t for t in by_text)


def test_apply_temporal_metadata_sets_retraction_for_past():
    classified = [
        ClassifiedClause(text="I used to play football", temporal_status="past"),
        ClassifiedClause(text="I play basketball", temporal_status="current"),
    ]
    facts = [
        _fact(
            relation="ENJOYS",
            object="football",
            fact_string="user enjoys football",
            source_span="I used to play football",
        ),
        _fact(
            relation="ENJOYS",
            object="basketball",
            fact_string="user enjoys basketball",
            source_span="switched to basketball two years ago",
        ),
    ]
    out = apply_temporal_metadata(facts, classified)
    past = next(f for f in out if f.object == "football")
    current = next(f for f in out if f.object == "basketball")
    assert past.temporal_status == "past" and past.retraction_signal is True
    assert current.temporal_status == "current" and current.retraction_signal is False


# --- Step 3: grounding ------------------------------------------------------
def test_validate_grounding_accepts_verbatim_span():
    text = TEST_INPUT
    fact = _fact(source_span="I live with two cats named Luna and Milo.", object="Luna")
    assert validate_grounding(fact, text) is True


def test_validate_grounding_rejects_hallucinated_span():
    fact = _fact(source_span="user lives in cats", object="cats")
    assert validate_grounding(fact, TEST_INPUT) is False


def test_ensure_source_spans_infers_from_object():
    facts = [_fact(source_span="", object="Luna", relation="HAS_PET", fact_string="user has pet Luna")]
    out = ensure_source_spans(facts, TEST_INPUT)
    assert "Luna" in out[0].source_span


# --- Step 4 & 5: gate_facts -------------------------------------------------
@pytest.mark.asyncio
async def test_gate_facts_rejects_low_confidence(db_session):
    session, episode_id = db_session
    fact = _fact(confidence=0.5, source_span="I am learning guitar since last month.")
    out = await gate_facts(
        session, "u_test", episode_id, [fact], source_text=TEST_INPUT
    )
    assert out == []


@pytest.mark.asyncio
async def test_gate_facts_rejects_ungrounded(db_session):
    session, episode_id = db_session
    fact = _fact(
        relation="LIVES_IN",
        object="cats",
        fact_string="user lives in cats",
        source_span="user lives in cats",
        confidence=0.95,
    )
    out = await gate_facts(
        session, "u_test", episode_id, [fact], source_text=TEST_INPUT
    )
    assert out == []


@pytest.mark.asyncio
async def test_gate_facts_rejects_lives_in_for_pets(db_session):
    session, episode_id = db_session
    fact = _fact(
        relation="LIVES_IN",
        object="cats",
        fact_string="user lives in cats",
        source_span="I live with two cats named Luna and Milo.",
        confidence=0.95,
    )
    assert validate_lives_in(fact) is False
    out = await gate_facts(
        session, "u_test", episode_id, [fact], source_text=TEST_INPUT
    )
    assert out == []


@pytest.mark.asyncio
async def test_gate_facts_accepts_has_pet_with_grounding(db_session):
    session, episode_id = db_session
    facts = [
        _fact(
            relation="HAS_PET",
            object="Luna",
            fact_string="user has pet Luna",
            source_span="I live with two cats named Luna and Milo.",
            confidence=0.95,
        ),
        _fact(
            relation="HAS_PET",
            object="Milo",
            fact_string="user has pet Milo",
            source_span="I live with two cats named Luna and Milo.",
            confidence=0.95,
        ),
    ]
    out = await gate_facts(
        session, "u_test", episode_id, facts, source_text=TEST_INPUT
    )
    assert {f.object for f in out} == {"Luna", "Milo"}
    for f in out:
        assert validate_grounding(f, TEST_INPUT)


@pytest.mark.asyncio
async def test_gate_facts_skips_future_facts(db_session):
    session, episode_id = db_session
    fact = _fact(
        relation="WORKS_AT",
        object="Google",
        source_span="I will join Google next month",
        temporal_status="future",
        confidence=0.95,
    )
    out = await gate_facts(
        session,
        "u_test",
        episode_id,
        [fact],
        source_text="I will join Google next month.",
    )
    assert out == []


# --- Full pipeline (mocked LLM) ---------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_football_cats_scenario(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(gliner_extractor, "extract", lambda content: [])

    async def _mock_llm(content, gliner_facts=None, classified=None):
        from mnemo.app.models.extraction import ExtractionResult

        return ExtractionResult(
            facts=[
                TripletFact(
                    subject="user",
                    relation="ENJOYS",
                    object="football",
                    fact_string="user enjoys football",
                    source_span="I used to play football",
                    confidence=0.92,
                    review_status=REVIEW_CONFIRMED,
                    source="llm",
                ),
                TripletFact(
                    subject="user",
                    relation="ENJOYS",
                    object="basketball",
                    fact_string="user enjoys basketball",
                    source_span="switched to basketball two years ago",
                    confidence=0.93,
                    review_status=REVIEW_CONFIRMED,
                    source="llm",
                ),
                TripletFact(
                    subject="user",
                    relation="LEARNING",
                    object="guitar",
                    fact_string="user learning guitar",
                    source_span="I am learning guitar since last month.",
                    confidence=0.94,
                    review_status=REVIEW_CONFIRMED,
                    source="llm",
                ),
                TripletFact(
                    subject="user",
                    relation="HAS_PET",
                    object="Luna",
                    fact_string="user has pet Luna",
                    source_span="I live with two cats named Luna and Milo.",
                    confidence=0.96,
                    review_status=REVIEW_CONFIRMED,
                    source="llm",
                ),
                TripletFact(
                    subject="user",
                    relation="HAS_PET",
                    object="Milo",
                    fact_string="user has pet Milo",
                    source_span="I live with two cats named Luna and Milo.",
                    confidence=0.96,
                    review_status=REVIEW_CONFIRMED,
                    source="llm",
                ),
            ]
        )

    monkeypatch.setattr(structured_extractor, "extract", _mock_llm)

    async def _classified(text: str):
        return [
            ClassifiedClause(text="I used to play football", temporal_status="past"),
            ClassifiedClause(text="switched to basketball two years ago", temporal_status="current"),
            ClassifiedClause(text="I am learning guitar since last month.", temporal_status="current"),
            ClassifiedClause(text="I live with two cats named Luna and Milo.", temporal_status="current"),
        ]

    monkeypatch.setattr("mnemo.app.services.extraction.pipeline.classify_clauses", _classified)
    result = await pipeline.extract_facts(TEST_INPUT, user_id="u1")
    facts, _ = apply_temporal_signals(TEST_INPUT, result.facts)

    football = next(f for f in facts if f.object == "football")
    basketball = next(f for f in facts if f.object == "basketball")
    guitar = [f for f in facts if f.object == "guitar"]
    pets = [f for f in facts if f.relation == "HAS_PET"]

    assert football.temporal_status == "past" and football.retraction_signal is True
    assert basketball.temporal_status == "current" and basketball.retraction_signal is False
    assert len(guitar) == 1
    assert {p.object for p in pets} == {"Luna", "Milo"}
    assert not any(f.relation == "LIVES_IN" for f in facts)
    for f in facts:
        assert validate_grounding(f, TEST_INPUT)
