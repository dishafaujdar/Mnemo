"""Deterministic NER and simple triplet extraction via spaCy (coding-context aware)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mnemo.app.models.extraction import TripletFact

if TYPE_CHECKING:
    pass

# Relation phrases that map to our relation types
RELATION_PATTERNS = [
    (r"\b(?:use|uses|using)\s+(?:the\s+)?([A-Za-z0-9+#]+)\b", "USES"),
    (r"\b(?:prefer|prefers|preferred)\s+(?:the\s+)?([A-Za-z0-9+#]+)\b", "PREFERS"),
    (r"\b(?:work|works|working)\s+(?:on|with)\s+(?:the\s+)?([A-Za-z0-9\s]+?)(?:\s+using|\s+with|\.|,|$)", "WORKS_ON"),
    (r"\b(?:build|building|built)\s+(?:a\s+)?([A-Za-z0-9\s]+?)(?:\s+with|\s+in|\.|,|$)", "BUILDING"),
    (r"\b(?:know|knows)\s+([A-Za-z0-9+#]+)\b", "KNOWS"),
    (r"\b(?:name|named?|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", "IS"),
    (r"\b(?:am|is|are)\s+(?:a\s+)?([a-z][a-z\s]+?)(?:\s+developer|\.|,|$)", "IS"),
    (r"\b(?:love|loves|loving)\s+([A-Za-z0-9+#]+)\b", "PREFERS"),
    (r"\b(?:hate|hates|dislike|dislikes)\s+([A-Za-z0-9+#]+)\b", "DISLIKES"),
    (r"\b(?:switched to|switch to)\s+([A-Za-z0-9+#]+)\b", "SWITCHED_TO"),
    (r"\b(?:switched from|used to use)\s+([A-Za-z0-9+#]+)\b", "SWITCHED_FROM"),
    (r"\b(?:goal|goals?)\s+(?:is|are)\s+(.+?)(?:\.|$)", "GOAL_IS"),
]

# Known tech entities for coding context (supplement NER)
TECH_ENTITIES = {
    "python", "go", "rust", "typescript", "javascript", "java", "c++", "ruby",
    "fastapi", "django", "react", "vue", "langchain", "next.js",
    "docker", "kubernetes", "git", "cursor", "vscode", "vim",
    "sqlite", "postgres", "redis", "qdrant",
}


def _normalize_obj(s: str) -> str:
    return s.strip().strip(".,")[:512]


def extract(content: str, spacy_nlp: object | None = None) -> list[TripletFact]:
    """
    Extract triplets from text using regex patterns (coding-assistant context).
    If spacy_nlp is provided, also add entities from NER as KNOWS or subject/object.
    """
    facts: list[TripletFact] = []
    content_lower = content.lower()
    seen: set[tuple[str, str, str]] = set()

    for pattern, relation in RELATION_PATTERNS:
        for m in re.finditer(pattern, content, re.IGNORECASE):
            obj = _normalize_obj(m.group(1))
            if not obj or len(obj) < 2:
                continue
            key = ("user", relation, obj.lower())
            if key in seen:
                continue
            seen.add(key)
            fact_str = f"User {relation.lower().replace('_', ' ')} {obj}"
            facts.append(
                TripletFact(
                    subject="user",
                    relation=relation,
                    object=obj,
                    fact_string=fact_str,
                    confidence=0.85,
                )
            )

    if spacy_nlp is not None:
        doc = spacy_nlp(content)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE") and ent.text.strip():
                obj = _normalize_obj(ent.text)
                key = ("user", "KNOWS", obj.lower())
                if key not in seen:
                    seen.add(key)
                    facts.append(
                        TripletFact(
                            subject="user",
                            relation="KNOWS",
                            object=obj,
                            fact_string=f"User knows {obj}",
                            confidence=0.7,
                        )
                    )
        for token in doc:
            if token.text.lower() in TECH_ENTITIES and token.head.dep_ in ("dobj", "attr", "pobj"):
                obj = _normalize_obj(token.text)
                key = ("user", "USES", obj.lower())
                if key not in seen:
                    seen.add(key)
                    facts.append(
                        TripletFact(
                            subject="user",
                            relation="USES",
                            object=obj,
                            fact_string=f"User uses {obj}",
                            confidence=0.75,
                        )
                    )

    return facts


def get_nlp(model_name: str):
    """Load spaCy model (caller should run: python -m spacy download en_core_web_sm)."""
    try:
        import spacy
        return spacy.load(model_name)
    except OSError:
        return None
