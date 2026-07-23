"""Soft-ontology manager: normalize relations via fuzzy matching, auto-expand.

Design goals:
- Never drop a fact just because its relation is unknown (fall back to MULTI).
- Handle typos, casing and phrasing via fuzzy matching against known aliases.
- Learn: fuzzy hits auto-register as aliases; frequently-seen unknown relations
  are auto-promoted into the canonical set.

Matching tiers returned by :meth:`OntologyManager.normalize`:
- ``CONFIRMED`` (score 1.0): exact alias / canonical match.
- ``FUZZY`` (score >= threshold): close match, alias auto-learned.
- ``UNKNOWN`` (score < threshold): kept, canonicalized, flagged for review.
- ``REJECT`` (score 0, no usable token): discard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mnemo.app.core.config import settings
from mnemo.app.services.ontology.seed import (
    DEFAULT_BEHAVIOR,
    SEED_ALIASES,
    SEED_BEHAVIOR,
    RelationBehavior,
)
from mnemo.app.services.ontology.canonical import is_blacklisted_relation

TIER_CONFIRMED = "confirmed"
TIER_FUZZY = "fuzzy"
TIER_UNKNOWN = "unknown"
TIER_REJECT = "reject"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _canonicalize(relation_raw: str) -> str:
    """Turn arbitrary phrasing into a CANONICAL_TOKEN (UPPER_SNAKE_CASE)."""
    cleaned = _NON_ALNUM.sub("_", relation_raw.strip().lower()).strip("_")
    return cleaned.upper()


def _alias_key(relation_raw: str) -> str:
    """Normalized lookup key for the alias map (spaces, single-spaced)."""
    return _NON_ALNUM.sub(" ", relation_raw.strip().lower()).strip()


def _similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity score between two short strings.

    Uses rapidfuzz when available; falls back to difflib otherwise.
    """
    try:  # optional dependency, much faster/better
        from rapidfuzz import fuzz

        return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b)) / 100.0
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio()


@dataclass
class MatchResult:
    """Outcome of normalizing a raw relation against the ontology."""

    relation: str  # canonical relation token (may be newly minted for UNKNOWN)
    relation_raw: str
    match_score: float
    tier: str  # one of TIER_*
    behavior: RelationBehavior

    @property
    def is_rejected(self) -> bool:
        return self.tier == TIER_REJECT


@dataclass
class _UnknownStat:
    count: int = 0
    conf_sum: float = 0.0

    @property
    def avg_confidence(self) -> float:
        return self.conf_sum / self.count if self.count else 0.0


@dataclass
class OntologyManager:
    """Runtime ontology. Seeded from :mod:`seed`, learns as it runs."""

    aliases: dict[str, str] = field(default_factory=lambda: dict(SEED_ALIASES))
    behavior: dict[str, RelationBehavior] = field(
        default_factory=lambda: dict(SEED_BEHAVIOR)
    )
    unknown_stats: dict[str, _UnknownStat] = field(default_factory=dict)
    learned_aliases: dict[str, str] = field(default_factory=dict)

    # -- lookups -----------------------------------------------------------
    def relations(self) -> set[str]:
        return set(self.behavior.keys())

    def behavior_for(self, relation: str) -> RelationBehavior:
        return self.behavior.get(relation.upper(), DEFAULT_BEHAVIOR)

    def is_singular(self, relation: str) -> bool:
        return self.behavior_for(relation) is RelationBehavior.SINGULAR

    def is_temporal(self, relation: str) -> bool:
        return self.behavior_for(relation) is RelationBehavior.TEMPORAL

    # -- normalization -----------------------------------------------------
    def normalize(self, relation_raw: str) -> MatchResult:
        """Sync alias lookup only (legacy). Prefer :meth:`normalize_async`."""
        return self._normalize_alias(relation_raw)

    async def normalize_async(self, relation_raw: str) -> MatchResult:
        """Map a raw relation via alias fast-path, then semantic matching."""
        alias_result = self._normalize_alias(relation_raw)
        if alias_result.tier in (TIER_CONFIRMED, TIER_FUZZY):
            if is_blacklisted_relation(alias_result.relation):
                return MatchResult(
                    alias_result.relation,
                    relation_raw,
                    0.0,
                    TIER_REJECT,
                    alias_result.behavior,
                )
            return alias_result

        from mnemo.app.services.ontology.semantic_match import semantic_normalize

        semantic = await semantic_normalize(relation_raw)
        if is_blacklisted_relation(semantic.relation):
            return MatchResult(
                semantic.relation,
                relation_raw,
                semantic.match_score,
                TIER_REJECT,
                semantic.behavior,
            )
        return semantic

    def _normalize_alias(self, relation_raw: str) -> MatchResult:
        """Map a raw relation string to a canonical relation + tier (alias/fuzzy)."""
        key = _alias_key(relation_raw)
        if not key:
            return MatchResult(
                relation="",
                relation_raw=relation_raw,
                match_score=0.0,
                tier=TIER_REJECT,
                behavior=DEFAULT_BEHAVIOR,
            )

        # 1. Exact alias match.
        if key in self.aliases:
            rel = self.aliases[key]
            return MatchResult(rel, relation_raw, 1.0, TIER_CONFIRMED, self.behavior_for(rel))

        # 2. Exact canonical match (e.g. "WORKS_AT" or "works_at").
        canonical = _canonicalize(relation_raw)
        if canonical in self.behavior:
            return MatchResult(
                canonical, relation_raw, 1.0, TIER_CONFIRMED, self.behavior_for(canonical)
            )

        # 3. Fuzzy match against known aliases and canonical tokens.
        best_rel, best_score = self._best_fuzzy(key, canonical)
        if best_rel is not None and best_score >= settings.ontology_fuzzy_threshold:
            # Auto-learn the alias so next time it's an exact hit.
            self.aliases[key] = best_rel
            self.learned_aliases[key] = best_rel
            return MatchResult(
                best_rel, relation_raw, best_score, TIER_FUZZY, self.behavior_for(best_rel)
            )

        # 4. Unknown: keep it (never discard), default to MULTI, flag for review.
        return MatchResult(
            canonical,
            relation_raw,
            round(best_score, 3),
            TIER_UNKNOWN,
            DEFAULT_BEHAVIOR,
        )

    def _best_fuzzy(self, key: str, canonical: str) -> tuple[str | None, float]:
        best_rel: str | None = None
        best_score = 0.0
        # Compare against alias phrasings.
        for alias, rel in self.aliases.items():
            score = _similarity(key, alias)
            if score > best_score:
                best_rel, best_score = rel, score
        # Compare against canonical tokens directly (helps "WorksAt" style input).
        canon_probe = canonical.lower().replace("_", " ")
        for rel in self.behavior:
            score = _similarity(canon_probe, rel.lower().replace("_", " "))
            if score > best_score:
                best_rel, best_score = rel, score
        return best_rel, best_score

    # -- auto-expansion ----------------------------------------------------
    def record_unknown(self, relation: str, confidence: float) -> bool:
        """Track an UNKNOWN relation sighting; auto-promote when it's common.

        Returns True if the relation was promoted into the canonical set.
        """
        rel = relation.upper()
        if not rel or rel in self.behavior:
            return False
        stat = self.unknown_stats.setdefault(rel, _UnknownStat())
        stat.count += 1
        stat.conf_sum += max(0.0, min(1.0, confidence))
        if (
            stat.count >= settings.ontology_auto_promote_count
            and stat.avg_confidence >= settings.ontology_auto_promote_confidence
        ):
            # Promote as MULTI (safest); an audit can reclassify later.
            self.behavior[rel] = DEFAULT_BEHAVIOR
            self.aliases[_alias_key(rel.replace("_", " "))] = rel
            self.unknown_stats.pop(rel, None)
            return True
        return False

    def learn_alias(self, alias: str, relation: str) -> None:
        key = _alias_key(alias)
        rel = relation.upper()
        if key and rel:
            self.aliases[key] = rel
            self.learned_aliases[key] = rel
            self.behavior.setdefault(rel, DEFAULT_BEHAVIOR)

    def pending_unknowns(self) -> dict[str, _UnknownStat]:
        """Snapshot of the unknown-relation audit queue."""
        return dict(self.unknown_stats)


_ontology: OntologyManager | None = None


def get_ontology() -> OntologyManager:
    """Process-wide ontology singleton."""
    global _ontology
    if _ontology is None:
        _ontology = OntologyManager()
    return _ontology
