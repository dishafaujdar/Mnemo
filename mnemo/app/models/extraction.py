"""Pydantic schemas for fact extraction (triplets)."""

from pydantic import BaseModel


class TripletFact(BaseModel):
    """A single extracted fact: subject --[relation]--> object."""

    subject: str
    relation: str
    object: str
    fact_string: str
    confidence: float = 1.0
