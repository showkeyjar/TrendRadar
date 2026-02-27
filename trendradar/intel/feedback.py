# coding=utf-8
"""
反馈学习状态管理
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_feedback_state(state_file: str) -> Dict:
    path = Path(state_file)
    if not path.exists():
        return {
            "keyword_adjustments": {},
            "source_adjustments": {},
            "added_keywords": [],
            "blocked_keywords": [],
            "events": [],
            "updated_at": _now_iso(),
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_feedback_state(state_file: str, state: Dict) -> None:
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _update_factor(factor: float, rating: int, step: float, min_v: float, max_v: float) -> float:
    delta = (rating - 3) * step
    next_factor = factor * (1.0 + delta)
    return max(min_v, min(max_v, next_factor))


def _unique_extend(base: List[str], values: List[str]) -> List[str]:
    merged = list(base)
    seen = {x for x in merged}
    for value in values:
        v = value.strip()
        if not v or v in seen:
            continue
        merged.append(v)
        seen.add(v)
    return merged


def record_feedback(
    state_file: str,
    rating: int,
    keywords: List[str] | None = None,
    source_names: List[str] | None = None,
    add_keywords: List[str] | None = None,
    block_keywords: List[str] | None = None,
    note: str = "",
) -> Dict:
    state = load_feedback_state(state_file)
    keywords = [k.strip().lower() for k in (keywords or []) if k.strip()]
    source_names = [s.strip().lower() for s in (source_names or []) if s.strip()]
    add_keywords = [k.strip().lower() for k in (add_keywords or []) if k.strip()]
    block_keywords = [k.strip().lower() for k in (block_keywords or []) if k.strip()]

    if rating < 1 or rating > 5:
        raise ValueError("rating 必须在 1~5")

    keyword_adjust = state.setdefault("keyword_adjustments", {})
    for keyword in keywords:
        old = float(keyword_adjust.get(keyword, 1.0))
        keyword_adjust[keyword] = _update_factor(old, rating, step=0.08, min_v=0.4, max_v=2.5)

    source_adjust = state.setdefault("source_adjustments", {})
    for source in source_names:
        old = float(source_adjust.get(source, 1.0))
        source_adjust[source] = _update_factor(old, rating, step=0.05, min_v=0.5, max_v=2.0)

    if add_keywords:
        state["added_keywords"] = _unique_extend(state.get("added_keywords", []), add_keywords)
    if block_keywords:
        state["blocked_keywords"] = _unique_extend(state.get("blocked_keywords", []), block_keywords)

    event = {
        "time": _now_iso(),
        "rating": rating,
        "keywords": keywords,
        "source_names": source_names,
        "add_keywords": add_keywords,
        "block_keywords": block_keywords,
        "note": note.strip(),
    }
    events = state.setdefault("events", [])
    events.append(event)
    if len(events) > 200:
        state["events"] = events[-200:]

    state["updated_at"] = _now_iso()
    save_feedback_state(state_file, state)
    return state

