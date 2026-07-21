"""Conflict slots: relations that compete for the same semantic 'seat'."""

from __future__ import annotations

# One active value per slot (cross-relation invalidation).
SINGULAR_SLOTS: dict[str, frozenset[str]] = {
    "role": frozenset({"IS", "HAS_ROLE"}),
    "employer": frozenset({"WORKS_AT"}),
    "residence": frozenset({"LIVES_IN"}),
    "origin": frozenset({"BORN_IN"}),
    "goal": frozenset({"GOAL_IS"}),
}

# Transition/event relations — describe a change, not durable state. Never store.
TRANSITION_RELATIONS: frozenset[str] = frozenset(
    {
        "SWITCHED_FROM",
        "SWITCHED_TO",
    }
)

# Past-tense employment; may be stored briefly but invalidated when WORKS_AT updates.
EMPLOYMENT_PAST_RELATIONS: frozenset[str] = frozenset({"WORKED_AT"})

EMPLOYMENT_CURRENT_RELATIONS: frozenset[str] = frozenset({"WORKS_AT"})

EMPLOYMENT_OBJECT_RELATIONS: frozenset[str] = frozenset(
    {"WORKS_AT", "WORKED_AT", "SWITCHED_FROM"}
)

LOCATION_OBJECT_RELATIONS: frozenset[str] = frozenset({"LIVES_IN", "LIVED_IN", "BORN_IN"})

SUPERSEDES: dict[str, frozenset[str]] = {
    "LIVES_IN": frozenset({"BORN_IN", "LIVED_IN"}),
    "WORKS_AT": frozenset({"WORKED_AT", "SWITCHED_FROM", "SWITCHED_TO"}),
}


def slot_for_relation(relation: str) -> str | None:
    rel = relation.upper()
    for slot, relations in SINGULAR_SLOTS.items():
        if rel in relations:
            return slot
    return None


def relations_in_slot(slot: str) -> frozenset[str]:
    return SINGULAR_SLOTS.get(slot, frozenset())


def supersedes(new_relation: str, old_relation: str) -> bool:
    new_r = new_relation.upper()
    old_r = old_relation.upper()
    return old_r in SUPERSEDES.get(new_r, frozenset())
