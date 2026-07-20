"""Soft ontology: relation behavior map, fuzzy matching, auto-expansion."""

from mnemo.app.services.ontology.manager import (
    MatchResult,
    OntologyManager,
    get_ontology,
)
from mnemo.app.services.ontology.seed import RelationBehavior

__all__ = [
    "MatchResult",
    "OntologyManager",
    "RelationBehavior",
    "get_ontology",
]
