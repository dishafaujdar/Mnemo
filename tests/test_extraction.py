"""Extraction tests: ontology matching, judge tiers, pipeline orchestration."""

import pytest

from mnemo.app.models.extraction import (
    REVIEW_CONFIRMED,
    REVIEW_FUZZY,
    REVIEW_PENDING,
    TripletFact,
)
from mnemo.app.services.extraction import gliner_extractor, pipeline, structured_extractor
from mnemo.app.services.extraction.judge import (
    ACTION_DISCARD,
    ACTION_REVIEW,
    ACTION_STORE,
    apply_judge,
    judge_action,
    review_status_for_tier,
)
from mnemo.app.services.extraction.pipeline import _merge_facts, _route_gliner
from mnemo.app.services.ontology.manager import (
    TIER_CONFIRMED,
    TIER_FUZZY,
    TIER_REJECT,
    TIER_UNKNOWN,
    OntologyManager,
)
from mnemo.app.services.ontology.seed import RelationBehavior


def _fact(relation="USES", obj="Python", conf=0.9, **kw):
    return TripletFact(
        subject="user",
        relation=relation,
        object=obj,
        fact_string=f"user {relation.lower()} {obj}",
        confidence=conf,
        **kw,
    )


# --- ontology ------------------------------------------------------------
def test_ontology_exact_alias_match():
    ont = OntologyManager()
    m = ont.normalize("works at")
    assert m.relation == "WORKS_AT"
    assert m.match_score == 1.0
    assert m.tier == TIER_CONFIRMED
    assert m.behavior is RelationBehavior.SINGULAR


def test_ontology_canonical_match():
    ont = OntologyManager()
    m = ont.normalize("USES")
    assert m.relation == "USES"
    assert m.tier == TIER_CONFIRMED
    assert m.behavior is RelationBehavior.MULTI


def test_ontology_fuzzy_match_learns_alias():
    ont = OntologyManager()
    m = ont.normalize("works att")
    assert m.relation == "WORKS_AT"
    assert m.tier == TIER_FUZZY
    assert 0.7 <= m.match_score < 1.0
    # Learned: a second call is now an exact hit.
    assert ont.normalize("works att").tier == TIER_CONFIRMED


def test_ontology_unknown_is_kept_not_rejected():
    ont = OntologyManager()
    m = ont.normalize("obsessed with")
    assert m.tier == TIER_UNKNOWN
    assert m.relation == "OBSESSED_WITH"
    assert m.behavior is RelationBehavior.MULTI  # safe fallback
    assert not m.is_rejected


def test_ontology_empty_is_rejected():
    ont = OntologyManager()
    assert ont.normalize("!!!").tier == TIER_REJECT
    assert ont.normalize("").is_rejected


def test_ontology_behavior_map():
    ont = OntologyManager()
    assert ont.is_singular("WORKS_AT")
    assert not ont.is_singular("USES")
    assert ont.is_temporal("VISITED")
    assert ont.behavior_for("UNSEEN_REL") is RelationBehavior.MULTI


def test_ontology_auto_promote():
    ont = OntologyManager()
    promoted = False
    for _ in range(20):
        promoted = ont.record_unknown("OBSESSED_WITH", 0.95) or promoted
    assert promoted
    assert "OBSESSED_WITH" in ont.relations()


# --- judge ---------------------------------------------------------------
def test_judge_action_thresholds():
    assert judge_action(0.9) == ACTION_STORE
    assert judge_action(0.5) == ACTION_REVIEW
    assert judge_action(0.2) == ACTION_DISCARD


def test_review_status_for_tier():
    assert review_status_for_tier(TIER_CONFIRMED) == REVIEW_CONFIRMED
    assert review_status_for_tier(TIER_FUZZY) == REVIEW_FUZZY
    assert review_status_for_tier(TIER_UNKNOWN) == REVIEW_PENDING


def test_apply_judge_filters_and_flags():
    keep = _fact(conf=0.9, review_status=REVIEW_CONFIRMED)
    review = _fact(obj="Go", conf=0.5, review_status=REVIEW_CONFIRMED)
    drop = _fact(obj="Rust", conf=0.1, review_status=REVIEW_CONFIRMED)
    out = apply_judge([keep, review, drop])
    objs = {f.object for f in out}
    assert "Python" in objs and "Go" in objs and "Rust" not in objs
    reviewed = next(f for f in out if f.object == "Go")
    assert reviewed.review_status == REVIEW_PENDING


# --- pipeline ------------------------------------------------------------
def test_route_gliner_splits_by_confidence():
    high = _fact(obj="Python", conf=0.92)
    low = _fact(obj="Go", conf=0.5)
    hi, lo = _route_gliner([high, low])
    assert hi == [high] and lo == [low]


def test_merge_facts_llm_supersedes_and_dedupes():
    g = _fact(obj="Python", conf=0.95, source="gliner")
    llm = _fact(obj="Python", conf=0.8, source="llm")
    merged = _merge_facts([g], [llm])
    assert len(merged) == 1
    assert merged[0].source == "llm"  # LLM validated against raw text, wins ties


async def test_extract_facts_high_conf_skips_llm(monkeypatch):
    high = _fact(obj="Python", conf=0.95, source="gliner")
    monkeypatch.setattr(gliner_extractor, "extract", lambda content: [high])

    async def _no_llm(content, gliner_facts=None):
        raise AssertionError("LLM should not be called for all-high-confidence facts")

    monkeypatch.setattr(structured_extractor, "extract", _no_llm)
    facts = await pipeline.extract_facts("I use Python", user_id="u1")
    assert [f.object for f in facts] == ["Python"]


async def test_extract_facts_low_conf_triggers_llm(monkeypatch):
    low = _fact(obj="Stripe", relation="WORKS_AT", conf=0.5, source="gliner")
    monkeypatch.setattr(gliner_extractor, "extract", lambda content: [low])

    async def _llm(content, gliner_facts=None):
        return [_fact(obj="Stripe", relation="WORKS_AT", conf=0.96, source="llm")]

    monkeypatch.setattr(structured_extractor, "extract", _llm)
    facts = await pipeline.extract_facts("I work for Stripe", user_id="u1")
    assert len(facts) == 1
    assert facts[0].source == "llm" and facts[0].confidence == 0.96


async def test_extract_facts_llm_fallback_keeps_gliner_when_llm_empty(monkeypatch):
    low = _fact(obj="Stripe", relation="WORKS_AT", conf=0.5, source="gliner")
    monkeypatch.setattr(gliner_extractor, "extract", lambda content: [low])

    async def _empty(content, gliner_facts=None):
        return []

    monkeypatch.setattr(structured_extractor, "extract", _empty)
    facts = await pipeline.extract_facts("I work for Stripe", user_id="u1")
    assert [f.object for f in facts] == ["Stripe"]  # not dropped
