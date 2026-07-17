from typing import List, Optional

from pydantic import BaseModel


class Pin(BaseModel):
    id: str
    title: str
    description: str
    tags: List[str]
    board: str
    image_url: str


class RecommendedPin(BaseModel):
    pin_id: str
    title: str
    board: str
    image_url: str
    score: float


class RecommendResponse(BaseModel):
    results: List[RecommendedPin]


class RecommendRequest(BaseModel):
    query: str
    k: Optional[int] = 5
