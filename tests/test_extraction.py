"""Extraction pipeline tests: spaCy and merge logic."""

import pytest

from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.extraction.pipeline import _average_confidence, _merge_facts, _needs_llm_pass
from mnemo.app.services.extraction.spacy_extractor import _is_locally_negated, extract as spacy_extract


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


def test_needs_llm_pass_low_confidence():
    facts = [
        TripletFact(subject="user", relation="USES", object="X", fact_string="User uses X", confidence=0.72),
        TripletFact(subject="user", relation="USES", object="Y", fact_string="User uses Y", confidence=0.73),
    ]
    assert _average_confidence(facts) < 0.78
    assert _needs_llm_pass(facts, "I use X and Y") is True


def test_needs_llm_pass_preference_language():
    two_facts = [
        TripletFact(subject="user", relation="USES", object="A", fact_string="User uses A", confidence=0.8),
        TripletFact(subject="user", relation="USES", object="B", fact_string="User uses B", confidence=0.8),
    ]
    assert _needs_llm_pass(two_facts, "I prefer Python over Go") is True


def test_needs_llm_pass_high_confidence_sufficient_facts():
    facts = [
        TripletFact(subject="user", relation="USES", object="Python", fact_string="User uses Python", confidence=0.9),
        TripletFact(subject="user", relation="USES", object="FastAPI", fact_string="User uses FastAPI", confidence=0.87),
        TripletFact(subject="user", relation="WORKS_ON", object="memory service", fact_string="User works on memory service", confidence=0.84),
    ]
    assert _average_confidence(facts) >= 0.78
    assert _needs_llm_pass(facts, "I use Python and FastAPI and work on a memory service.") is False


def test_spacy_avoids_noisy_is_extraction():
    facts = spacy_extract("I am working on a new project which is for the memory context.", spacy_nlp=None)
    assert not any(f.relation == "IS" and "memory context" in f.object.lower() for f in facts)


def test_local_negation_window_is_scoped():
    assert _is_locally_negated("I do not use React anymore.", start=13, end=18) is True
    assert _is_locally_negated("I do not use React anymore, but I use FastAPI now.", start=39, end=46) is False


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
