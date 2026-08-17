"""Seed ontology: canonical relations, their storage behavior, and aliases.

The ontology is intentionally *soft*: unknown relations are never dropped, they
fall back to MULTI-value storage and are flagged for later audit. Frequently
seen unknown relations can be auto-promoted (see ``manager.OntologyManager``).
"""

from __future__ import annotations

from enum import Enum


class RelationBehavior(str, Enum):
    """How the conflict resolver treats a relation when a new fact arrives."""

    SINGULAR = "singular"  # one current value: invalidate old, insert new
    MULTI = "multi"  # many coexisting values: keep old + new
    TEMPORAL = "temporal"  # history: keep every value, never invalidate


# Canonical relations grouped by behavior.
SINGULAR_RELATIONS: frozenset[str] = frozenset(
    {
        "IS",
        "WORKS_AT",
        "LIVES_IN",
        "HAS_ROLE",
        "GOAL_IS",
        "BORN_IN",
        "SWITCHED_TO",
        "STUDIES_AT",
    }
)

MULTI_RELATIONS: frozenset[str] = frozenset(
    {
        "USES",
        "PREFERS",
        "KNOWS",
        "LEARNING",
        "HAS_SKILL",
        "SPEAKS",
        "INTERESTED_IN",
        "ENJOYS",
        "DISLIKES",
        "SWITCHED_FROM",
        "WORKS_ON",
        "BUILDING",
        "STRUGGLES_WITH",
        "HAS",
        "HAS_PET",
        "PLAYS",
    }
)

TEMPORAL_RELATIONS: frozenset[str] = frozenset(
    {
        "LIVED_IN",
        "WORKED_AT",
        "STUDIED_AT",
        "VISITED",
        "TRAVELED_TO",
        "MET",
        "PLAYED",
    }
)


def _behavior_map() -> dict[str, RelationBehavior]:
    mapping: dict[str, RelationBehavior] = {}
    for rel in SINGULAR_RELATIONS:
        mapping[rel] = RelationBehavior.SINGULAR
    for rel in MULTI_RELATIONS:
        mapping[rel] = RelationBehavior.MULTI
    for rel in TEMPORAL_RELATIONS:
        mapping[rel] = RelationBehavior.TEMPORAL
    return mapping


# relation -> behavior (seed; the manager may extend this at runtime).
SEED_BEHAVIOR: dict[str, RelationBehavior] = _behavior_map()

# All canonical relation names.
SEED_RELATIONS: frozenset[str] = frozenset(SEED_BEHAVIOR.keys())

# Unknown relations default to the safest behavior: keep everything.
DEFAULT_BEHAVIOR: RelationBehavior = RelationBehavior.MULTI


# Human phrasing -> canonical relation. Lower-cased keys; matched exactly first,
# then fuzzily. The manager auto-learns new aliases as it sees them.
SEED_ALIASES: dict[str, str] = {
    # WORKS_AT
    "works at": "WORKS_AT",
    "work at": "WORKS_AT",
    "employed by": "WORKS_AT",
    "job at": "WORKS_AT",
    "works for": "WORKS_AT",
    "work for": "WORKS_AT",
    # LIVES_IN
    "lives in": "LIVES_IN",
    "live in": "LIVES_IN",
    "resides in": "LIVES_IN",
    "based in": "LIVES_IN",
    "moved to": "LIVES_IN",
    # USES
    "uses": "USES",
    "use": "USES",
    "works with": "USES",
    "codes in": "USES",
    "coding in": "USES",
    # PREFERS
    "prefers": "PREFERS",
    "prefer": "PREFERS",
    "likes": "PREFERS",
    "like": "PREFERS",
    "loves": "PREFERS",
    "favorite": "PREFERS",
    # DISLIKES
    "dislikes": "DISLIKES",
    "hates": "DISLIKES",
    # LEARNING
    "learning": "LEARNING",
    "studying": "LEARNING",
    "picking up": "LEARNING",
    # KNOWS / SKILLS
    "knows": "KNOWS",
    "has skill": "HAS_SKILL",
    "skilled in": "HAS_SKILL",
    "speaks": "SPEAKS",
    # ROLE / IDENTITY
    "is": "IS",
    "is a": "IS",
    "is an": "IS",
    "has role": "HAS_ROLE",
    "role is": "HAS_ROLE",
    "works as": "HAS_ROLE",
    # GOAL
    "goal is": "GOAL_IS",
    "goal": "GOAL_IS",
    "wants to": "GOAL_IS",
    # STUDY / ORIGIN
    "studies at": "STUDIES_AT",
    "studying at": "STUDIES_AT",
    "born in": "BORN_IN",
    # SWITCH
    "switched to": "SWITCHED_TO",
    "moved to stack": "SWITCHED_TO",
    "switched from": "SWITCHED_FROM",
    # INTEREST
    "interested in": "INTERESTED_IN",
    "enjoys": "ENJOYS",
    # PROJECT
    "works on": "WORKS_ON",
    "working on": "WORKS_ON",
    "building": "BUILDING",
    "struggles with": "STRUGGLES_WITH",
    # TEMPORAL
    "lived in": "LIVED_IN",
    "worked at": "WORKED_AT",
    "studied at": "STUDIED_AT",
    "visited": "VISITED",
    "traveled to": "TRAVELED_TO",
    "met": "MET",
    "played": "PLAYED",
    "used to play": "PLAYED",
    # SPORTS / ACTIVITIES
    "plays": "PLAYS",
    "play": "PLAYS",
    "playing": "PLAYS",
    # pets
    "has pet": "HAS_PET",
    "has pets": "HAS_PET",
    "live with": "HAS_PET",
    "lives with": "HAS_PET",
}


# Entity labels for GLiNER2 NER, and relation labels (canonical, lower-cased for
# the model). GLiNER emits the label back, which we normalize via the aliases.
GLINER_ENTITY_LABELS: list[str] = [
    "person",
    "organization",
    "location",
    "job_title",
    "technology",
    "skill",
    "hobby",
    "goal",
    "preference",
    "event",
    "date",
    "time",
]

# Relation labels handed to GLiNER's relation extraction, phrased naturally so
# the model recognizes them; each maps back through SEED_ALIASES.
GLINER_RELATION_LABELS: list[str] = [
    "works at",
    "lives in",
    "uses",
    "prefers",
    "dislikes",
    "learning",
    "knows",
    "speaks",
    "is",
    "has role",
    "goal is",
    "studies at",
    "born in",
    "switched to",
    "switched from",
    "interested in",
    "works on",
    "building",
    "has pet",
    "plays",
    "played",
]
