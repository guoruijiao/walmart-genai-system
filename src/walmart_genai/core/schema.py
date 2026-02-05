from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["product_info", "delivery", "return", "store_info", "other"]
NextAction = Literal["respond", "ask_clarification", "handoff_human"]


class Citation(BaseModel):
    source: str
    quote: str


class Entities(BaseModel):
    product_id: str | None = None
    store_id: str | None = None


class GenAIResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    intent: Intent
    entities: Entities = Field(default_factory=Entities)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    next_action: NextAction = "respond"
