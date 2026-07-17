"""LLM query parsing with deterministic offline fallback."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, replace
from typing import Any, Dict, Optional, Tuple

from geogenie.core.types import StructuredQuery

# Last successful / failed parse for GET /debug/last_parse
_LAST_PARSE: Dict[str, Any] = {}


class ParseError(Exception):
    def __init__(self, message: str, raw_llm_output: Any = None):
        super().__init__(message)
        self.raw_llm_output = raw_llm_output
        self.code = "PARSE_FAILED"


def get_last_parse() -> Dict[str, Any]:
    return dict(_LAST_PARSE)


def _set_last(payload: Dict[str, Any]) -> None:
    global _LAST_PARSE
    _LAST_PARSE = payload


def validate(sq: StructuredQuery) -> StructuredQuery:
    """Clamp radius and whitelist filters — never invent a silent default radius."""
    if sq.radius_m is None or sq.radius_m <= 0:
        raise ParseError("radius_m must be a positive number", raw_llm_output=asdict(sq))
    radius = max(50.0, min(float(sq.radius_m), 5000.0))
    allowed_noise = {None, "quiet", "moderate", "loud", "very_loud"}
    filters = dict(sq.filters or {})
    if "noise_level" in filters and filters["noise_level"] not in allowed_noise:
        raise ParseError(
            f"noise_level not in whitelist: {filters['noise_level']}",
            raw_llm_output=asdict(sq),
        )
    if "accessible" in filters and filters["accessible"] is not None:
        filters["accessible"] = bool(filters["accessible"])
    sort_by = sq.sort_by if sq.sort_by in {"distance", "name"} else "distance"
    return replace(sq, radius_m=radius, filters=filters, sort_by=sort_by)


_MINUTE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:min(?:ute)?s?|m\b)", re.I
)
_CATEGORY_KEYWORDS = {
    "cafe": "cafe",
    "coffee": "cafe",
    "restaurant": "restaurant",
    "bar": "bar",
    "park": "park",
    "grocery": "grocery",
    "pharmacy": "pharmacy",
    "library": "library",
    "gym": "gym",
    "museum": "museum",
    "retail": "retail",
}


def _heuristic_parse(
    text: str, origin_lon: float, origin_lat: float
) -> StructuredQuery:
    text_l = text.lower().strip()
    if not text_l:
        raise ParseError("empty query", raw_llm_output=text)

    m = _MINUTE_RE.search(text_l)
    if m:
        minutes = float(m.group(1))
        radius_m = minutes * 80.0  # WALK_SPEED_M_PER_MIN
    elif "walk" in text_l or "near" in text_l or "nearby" in text_l:
        minutes = 10.0
        radius_m = 800.0
    else:
        # Ambiguous — refuse rather than guess [VR §3.2 class]
        raise ParseError(
            "could not determine walk time or radius from query",
            raw_llm_output=text,
        )

    filters: Dict[str, Any] = {}
    if any(w in text_l for w in ("accessible", "wheelchair", "step-free", "step free")):
        filters["accessible"] = True
    if any(w in text_l for w in ("quiet", "not loud", "peaceful")):
        filters["noise_level"] = "quiet"
    if "not loud" in text_l or "isn't loud" in text_l:
        filters["noise_level"] = "quiet"

    category = None
    for kw, cat in _CATEGORY_KEYWORDS.items():
        if kw in text_l:
            category = cat
            filters["category"] = cat
            break

    return StructuredQuery(
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        radius_m=radius_m,
        minutes=minutes,
        filters=filters,
        sort_by="distance",
        category=category,
    )


def _llm_parse(
    text: str, origin_lon: float, origin_lat: float
) -> Optional[StructuredQuery]:
    """Try OpenAI/Anthropic structured parse; return None if unavailable."""
    schema_hint = {
        "radius_m": "number",
        "minutes": "number",
        "filters": {"accessible": "bool", "noise_level": "string", "category": "string"},
        "sort_by": "distance|name",
    }
    system = (
        "Parse the user location query into JSON matching this schema. "
        f"{json.dumps(schema_hint)}. "
        "If walk time is given, set minutes and radius_m = minutes * 80. "
        "If you cannot determine radius/minutes, return {\"error\": \"PARSE_FAILED\"}."
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            if data.get("error"):
                raise ParseError("LLM refused parse", raw_llm_output=data)
            return StructuredQuery(
                origin_lon=origin_lon,
                origin_lat=origin_lat,
                radius_m=float(data["radius_m"]),
                minutes=float(data["minutes"]) if data.get("minutes") is not None else None,
                filters=data.get("filters") or {},
                sort_by=data.get("sort_by") or "distance",
                category=(data.get("filters") or {}).get("category"),
            )
        except ParseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ParseError(f"LLM parse failed: {exc}", raw_llm_output=str(exc)) from exc

    anth_key = os.environ.get("ANTHROPIC_API_KEY")
    if anth_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=anth_key)
            msg = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=512,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": text}],
            )
            raw = msg.content[0].text
            # Extract JSON object
            start, end = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[start : end + 1])
            if data.get("error"):
                raise ParseError("LLM refused parse", raw_llm_output=data)
            return StructuredQuery(
                origin_lon=origin_lon,
                origin_lat=origin_lat,
                radius_m=float(data["radius_m"]),
                minutes=float(data["minutes"]) if data.get("minutes") is not None else None,
                filters=data.get("filters") or {},
                sort_by=data.get("sort_by") or "distance",
                category=(data.get("filters") or {}).get("category"),
            )
        except ParseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ParseError(f"LLM parse failed: {exc}", raw_llm_output=str(exc)) from exc

    return None


def parse_query(
    text: str,
    origin_lon: float,
    origin_lat: float,
    *,
    use_llm: bool = True,
) -> StructuredQuery:
    """Parse free text into StructuredQuery. Raises ParseError on failure."""
    try:
        sq = None
        if use_llm:
            sq = _llm_parse(text, origin_lon, origin_lat)
        if sq is None:
            sq = _heuristic_parse(text, origin_lon, origin_lat)
        sq = validate(sq)
        _set_last({"ok": True, "query": text, "parsed": asdict(sq)})
        return sq
    except ParseError as exc:
        _set_last(
            {
                "ok": False,
                "query": text,
                "error": str(exc),
                "raw_llm_output": exc.raw_llm_output,
            }
        )
        raise
