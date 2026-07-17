"""Pydantic request/response schemas (mirrors core.types)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OriginSchema(BaseModel):
    lat: float
    lon: float


class SearchRequest(BaseModel):
    query: str
    origin: OriginSchema


class StructuredSearchRequest(BaseModel):
    origin: OriginSchema
    radius_m: float = Field(..., gt=0)
    minutes: Optional[float] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    sort_by: str = "distance"
    category: Optional[str] = None


class POIResult(BaseModel):
    id: int
    lon: float
    lat: float
    name: Optional[str] = None
    category: Optional[str] = None
    accessible: Optional[bool] = None
    noise_level: Optional[str] = None
    distance_m: float
    explanation: str = ""


class SearchResponse(BaseModel):
    results: List[POIResult]
    query_parsed: Dict[str, Any]
    reach_ring: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    raw_llm_output: Any = None
