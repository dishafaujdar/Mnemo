"""Deterministic triplet extraction via regex and light spaCy dependency cues."""

from __future__ import annotations

import re

from mnemo.app.models.extraction import TripletFact

FIRST_PERSON_PATTERN = re.compile(r"\b(i|i'm|i’ve|i've|my|me|we|we're|we’ve|we've|our)\b", re.IGNORECASE)
NEGATION_PATTERN = re.compile(r"\b(?:not|never|no longer|don't|dont|didn't|didnt|isn't|isnt|aren't|arent)\b", re.IGNORECASE)
NEGATION_TERMS = {"not", "never", "no", "don't", "dont", "didn't", "didnt", "isn't", "isnt", "aren't", "arent"}
IS_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "for",
    "with",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "from",
    "about",
    "which",
    "that",
    "who",
    "is",
    "are",
    "was",
    "were",
}

RELATION_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"\b(?:use|uses|using)\s+(?:the\s+)?([A-Za-z0-9.+#-]+)\b", re.IGNORECASE), "USES", 0.83),
    (re.compile(r"\b(?:prefer|prefers|preferred)\s+(?:the\s+)?([A-Za-z0-9.+#-]+)\b", re.IGNORECASE), "PREFERS", 0.86),
    (re.compile(r"\b(?:love|loves|loving)\s+([A-Za-z0-9.+#-]+)\b", re.IGNORECASE), "PREFERS", 0.82),
    (re.compile(r"\b(?:hate|hates|dislike|dislikes)\s+([A-Za-z0-9.+#-]+)\b", re.IGNORECASE), "DISLIKES", 0.84),
    (
        re.compile(
            r"\b(?:work|works|working)\s+(?:on|with)\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9\s+#./-]{1,120}?)"
            r"(?:(?:\s+using|\s+with|\s+for)|[.,!?]|$)",
            re.IGNORECASE,
        ),
        "WORKS_ON",
        0.8,
    ),
    (
        re.compile(
            r"\b(?:build|building|built)\s+(?:a\s+|an\s+|the\s+)?([A-Za-z0-9][A-Za-z0-9\s+#./-]{1,120}?)"
            r"(?:(?:\s+with|\s+in|\s+for)|[.,!?]|$)",
            re.IGNORECASE,
        ),
        "BUILDING",
        0.82,
    ),
    (re.compile(r"\b(?:learn|learned|learning)\s+([A-Za-z0-9.+#-]+)\b", re.IGNORECASE), "LEARNED", 0.8),
    (
        re.compile(r"\b(?:struggle|struggles|struggling)\s+with\s+([A-Za-z0-9][A-Za-z0-9\s+#./-]{1,80})\b", re.IGNORECASE),
        "STRUGGLES_WITH",
        0.84,
    ),
    (re.compile(r"\b(?:work|works|working)\s+at\s+([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)", re.IGNORECASE), "WORKS_AT", 0.88),
    (re.compile(r"\b(?:switched to|switch to|moved to|move to)\s+([A-Za-z0-9.+#-]+)\b", re.IGNORECASE), "SWITCHED_TO", 0.88),
    (re.compile(r"\b(?:switched from|used to use|moved from|move from)\s+([A-Za-z0-9.+#-]+)\b", re.IGNORECASE), "SWITCHED_FROM", 0.88),
    (re.compile(r"\b(?:goal|goals?)\s+(?:is|are)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE), "GOAL_IS", 0.8),
    (re.compile(r"\b(?:call me|named?|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", re.IGNORECASE), "IS", 0.93),
    (re.compile(r"\b(?:i am|i'm|im)\s+(?:a\s+|an\s+)?([a-z][a-z\s-]{1,60}?)(?:[.,!?]|$)", re.IGNORECASE), "IS", 0.72),
]

TECH_ENTITIES = {
    "python": "Python",
    "go": "Go",
    "rust": "Rust",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "java": "Java",
    "c++": "C++",
    "ruby": "Ruby",
    "fastapi": "FastAPI",
    "django": "Django",
    "react": "React",
    "vue": "Vue",
    "langchain": "LangChain",
    "next.js": "Next.js",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "git": "Git",
    "cursor": "Cursor",
    "vscode": "VSCode",
    "vim": "Vim",
    "sqlite": "SQLite",
    "postgres": "Postgres",
    "postgresql": "Postgres",
    "redis": "Redis",
    "qdrant": "Qdrant",
}

LOW_SIGNAL_IS_OBJECTS = {
    "for the memory context",
    "for memory context",
    "working on",
    "new project",
    "a new project",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_obj(value: str) -> str:
    value = _normalize_whitespace(value.strip(" \t\r\n.,;:!?"))
    value = re.sub(r"^(?:a|an|the)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:which|that|who)\s+(?:is|are|was|were)\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:is|are|was|were)\s*$", "", value, flags=re.IGNORECASE)
    canonical = TECH_ENTITIES.get(value.lower())
    return (canonical or value)[:512]


def _looks_valid_object(value: str, relation: str) -> bool:
    if not value or len(value) < 2:
        return False
    if value.lower() in LOW_SIGNAL_IS_OBJECTS:
        return False
    if relation == "IS":
        tokens = re.findall(r"[A-Za-z]+", value.lower())
        if not tokens:
            return False
        banned_prefixes = ("for ", "with ", "using ", "working ")
        if value.lower().startswith(banned_prefixes):
            return False
        if all(token in IS_STOPWORDS for token in tokens):
            return False
        content_tokens = [token for token in tokens if token not in IS_STOPWORDS]
        if not content_tokens:
            return False
        if len(content_tokens) == 1 and content_tokens[0] in {"project", "context", "thing"}:
            return False
    return True


def _is_locally_negated(content: str, start: int | None = None, end: int | None = None) -> bool:
    if start is None or end is None:
        return bool(NEGATION_PATTERN.search(content))
    window_start = max(0, start - 32)
    window_end = min(len(content), end + 16)
    local_window = content[window_start:window_end].lower()
    if "no longer" in local_window:
        return True
    terms = re.findall(r"[a-z']+", local_window)
    return any(term in NEGATION_TERMS for term in terms[-6:])


def _confidence_for(
    content: str,
    relation: str,
    obj: str,
    base_confidence: float,
    start: int | None = None,
    end: int | None = None,
) -> float:
    confidence = base_confidence
    if FIRST_PERSON_PATTERN.search(content):
        confidence += 0.05
    if obj.lower() in TECH_ENTITIES:
        confidence += 0.04
    if relation in {"WORKS_AT", "SWITCHED_TO", "SWITCHED_FROM"}:
        confidence += 0.03
    if relation == "IS" and len(obj.split()) > 4:
        confidence -= 0.18
    if _is_locally_negated(content, start, end) and relation not in {"DISLIKES", "SWITCHED_FROM"}:
        confidence -= 0.08
    return max(0.0, min(0.98, confidence))


def _append_fact(
    facts: list[TripletFact],
    seen: set[tuple[str, str, str]],
    content: str,
    relation: str,
    obj: str,
    base_confidence: float,
    start: int | None = None,
    end: int | None = None,
) -> None:
    obj = _normalize_obj(obj)
    if not _looks_valid_object(obj, relation):
        return
    key = ("user", relation, obj.lower())
    if key in seen:
        return
    seen.add(key)
    facts.append(
        TripletFact(
            subject="user",
            relation=relation,
            object=obj,
            fact_string=f"User {relation.lower().replace('_', ' ')} {obj}",
            confidence=_confidence_for(content, relation, obj, base_confidence, start=start, end=end),
        )
    )


def _extract_regex_facts(content: str, facts: list[TripletFact], seen: set[tuple[str, str, str]]) -> None:
    for pattern, relation, base_confidence in RELATION_PATTERNS:
        for match in pattern.finditer(content):
            start, end = match.span(1)
            _append_fact(facts, seen, content, relation, match.group(1), base_confidence, start=start, end=end)


def _governing_lemma(token) -> str:
    current = token
    for _ in range(3):
        head = current.head
        if head == current:
            break
        lemma = head.lemma_.lower()
        if lemma and lemma != token.lemma_.lower():
            return lemma
        current = head
    return token.head.lemma_.lower()


def _extract_spacy_facts(content: str, spacy_nlp: object, facts: list[TripletFact], seen: set[tuple[str, str, str]]) -> None:
    doc = spacy_nlp(content)
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        sent_lower = sent_text.lower()

        for token in sent:
            token_lower = token.text.lower()
            if token_lower not in TECH_ENTITIES:
                continue
            if token.dep_ not in {"dobj", "attr", "pobj", "conj"}:
                continue
            governing_lemma = _governing_lemma(token)
            if governing_lemma in {"prefer", "like", "love"}:
                relation = "PREFERS"
            elif governing_lemma in {"hate", "dislike"}:
                relation = "DISLIKES"
            elif governing_lemma in {"use", "work", "build"}:
                relation = "USES"
            else:
                continue
            _append_fact(
                facts,
                seen,
                sent_text,
                relation,
                token.text,
                0.76,
                start=token.idx - sent.start_char,
                end=token.idx - sent.start_char + len(token.text),
            )

        root = sent.root
        if root.lemma_ == "work" and " at " in f" {sent_lower} ":
            for ent in sent.ents:
                if ent.label_ == "ORG":
                    _append_fact(
                        facts,
                        seen,
                        sent_text,
                        "WORKS_AT",
                        ent.text,
                        0.84,
                        start=ent.start_char - sent.start_char,
                        end=ent.end_char - sent.start_char,
                    )
        if root.lemma_ == "be" and FIRST_PERSON_PATTERN.search(sent_text):
            attr_tokens = [tok for tok in sent if tok.dep_ in {"attr", "acomp", "oprd"}]
            for tok in attr_tokens:
                span = doc[tok.left_edge.i : tok.right_edge.i + 1]
                _append_fact(
                    facts,
                    seen,
                    sent_text,
                    "IS",
                    span.text,
                    0.68,
                    start=span.start_char - sent.start_char,
                    end=span.end_char - sent.start_char,
                )


def extract(content: str, spacy_nlp: object | None = None) -> list[TripletFact]:
    """Extract triplets from text using regex rules and light spaCy parsing."""
    facts: list[TripletFact] = []
    seen: set[tuple[str, str, str]] = set()

    _extract_regex_facts(content, facts, seen)

    if spacy_nlp is not None:
        _extract_spacy_facts(content, spacy_nlp, facts, seen)

    return facts


def get_nlp(model_name: str):
    """Load spaCy model (caller should run: python -m spacy download en_core_web_sm)."""
    try:
        import spacy

        return spacy.load(model_name)
    except OSError:
        return None
