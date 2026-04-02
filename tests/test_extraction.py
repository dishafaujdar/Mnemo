"""Extraction pipeline tests: spaCy and merge logic."""

import pytest

from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.extraction.pipeline import _needs_llm_pass, _merge_facts
from mnemo.app.services.extraction.spacy_extractor import extract as spacy_extract


def test_spacy_extract_prefers():
    facts = spacy_extract("I prefer Python for backend", spacy_nlp=None)
    assert len(facts) >= 1
    assert any(f.relation == "PREFERS" and "Python" in f.object for f in facts)


def test_spacy_extract_uses():
    facts = spacy_extract("I use FastAPI and Docker", spacy_nlp=None)
    assert any(f.relation == "USES" for f in facts)


def test_needs_llm_pass_low_facts():
    assert _needs_llm_pass([], "I love TypeScript") is True
    assert _needs_llm_pass([TripletFact(subject="u", relation="USES", object="x", fact_string="u uses x", confidence=0.9)], "hello") is True


def test_needs_llm_pass_preference_language():
    two_facts = [
        TripletFact(subject="user", relation="USES", object="A", fact_string="User uses A", confidence=0.8),
        TripletFact(subject="user", relation="USES", object="B", fact_string="User uses B", confidence=0.8),
    ]
    assert _needs_llm_pass(two_facts, "I prefer Python over Go") is True


def test_merge_facts_dedupe():
    a = TripletFact(subject="user", relation="PREFERS", object="Python", fact_string="User prefers Python", confidence=0.9)
    b = TripletFact(subject="user", relation="PREFERS", object="Python", fact_string="User prefers Python", confidence=0.7)
    merged = _merge_facts([a], [b])
    assert len(merged) == 1
    assert merged[0].confidence == 0.9


def test_merge_facts_keep_higher_confidence():
    a = TripletFact(subject="user", relation="USES", object="X", fact_string="User uses X", confidence=0.7)
    b = TripletFact(subject="user", relation="USES", object="X", fact_string="User uses X", confidence=0.95)
    merged = _merge_facts([a], [b])
    assert len(merged) == 1
    assert merged[0].confidence == 0.95
