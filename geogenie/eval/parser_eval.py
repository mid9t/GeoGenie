"""LLM / heuristic parser evaluation against labeled queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from geogenie.genai.query_parser import ParseError, parse_query

HERE = Path(__file__).resolve().parent


def load_labeled(path: Path | None = None) -> List[Dict[str, Any]]:
    p = path or HERE / "labeled_queries.json"
    return json.loads(p.read_text())


def field_match(got: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, bool]:
    radius_ok = abs(got["radius_m"] - exp["radius_m"]) / max(exp["radius_m"], 1) <= 0.10
    filters_ok = got.get("filters") == exp.get("filters")
    sort_ok = got.get("sort_by", "distance") == exp.get("sort_by", "distance")
    return {"radius": radius_ok, "filters": filters_ok, "sort": sort_ok}


def evaluate(path: Path | None = None) -> Dict[str, Any]:
    labeled = load_labeled(path)
    per_field = {"radius": 0, "filters": 0, "sort": 0}
    full = 0
    failures = []
    for item in labeled:
        text = item["text"]
        origin = item["origin"]
        expected = item["expected"]
        expect_fail = item.get("expect_fail", False)
        try:
            sq = parse_query(
                text, origin["lon"], origin["lat"], use_llm=False
            )
            if expect_fail:
                failures.append({"text": text, "taxonomy": "should_have_refused"})
                continue
            got = {
                "radius_m": sq.radius_m,
                "filters": sq.filters,
                "sort_by": sq.sort_by,
            }
            m = field_match(got, expected)
            for k, v in m.items():
                if v:
                    per_field[k] += 1
            if all(m.values()):
                full += 1
            else:
                taxonomy = "wrong-field"
                if not m["filters"]:
                    taxonomy = "hallucinated-filter" if got["filters"] else "wrong-field"
                failures.append({"text": text, "taxonomy": taxonomy, "got": got, "expected": expected})
        except ParseError:
            if expect_fail:
                full += 1
                per_field["radius"] += 1
                per_field["filters"] += 1
                per_field["sort"] += 1
            else:
                failures.append({"text": text, "taxonomy": "refused"})

    n = len(labeled)
    return {
        "n": n,
        "per_field_accuracy": {k: v / n for k, v in per_field.items()},
        "full_query_accuracy": full / n,
        "failures": failures,
    }


def main() -> None:
    report = evaluate()
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    print(f"failures: {len(report['failures'])}")


if __name__ == "__main__":
    main()
