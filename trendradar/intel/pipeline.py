# coding=utf-8
"""
行业情报打分与摘要生成
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from trendradar.core import load_config
from trendradar.context import AppContext
from trendradar.storage.base import NewsItem, RSSItem

from .feedback import load_feedback_state
from .models import IntelItem, IntelProfile, IntelResult
from .profile import load_intel_profile


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _term_hit(text: str, term: str) -> bool:
    """
    Match a term against normalized text.
    Supports compound terms like "a+b" which require all parts to exist.
    """
    t = _norm(term)
    if not t:
        return False
    if "+" in t:
        parts = [p.strip() for p in t.split("+") if p.strip()]
        return bool(parts) and all(p in text for p in parts)
    return t in text


def _latest_data_date(db_dir: str) -> str | None:
    path = Path(db_dir)
    if not path.exists():
        return None
    candidates = sorted([p.stem for p in path.glob("*.db") if p.stem], reverse=True)
    return candidates[0] if candidates else None


def _flatten_news(ctx: AppContext, date: str | None = None) -> List[IntelItem]:
    data = ctx.get_storage_manager().get_today_all_data(date=date)
    if not data:
        return []
    rows: List[IntelItem] = []
    for source_id, items in data.items.items():
        source_name = data.id_to_name.get(source_id, source_id)
        for item in items:
            if not isinstance(item, NewsItem):
                continue
            rows.append(
                IntelItem(
                    title=item.title,
                    source_id=source_id,
                    source_name=source_name,
                    url=item.url or item.mobile_url,
                    rank=item.rank,
                    item_type="hotlist",
                )
            )
    return rows


def _flatten_rss(ctx: AppContext, date: str | None = None) -> List[IntelItem]:
    data = ctx.get_storage_manager().get_rss_data(date=date)
    if not data:
        return []
    rows: List[IntelItem] = []
    for feed_id, items in data.items.items():
        feed_name = data.id_to_name.get(feed_id, feed_id)
        for item in items:
            if not isinstance(item, RSSItem):
                continue
            rows.append(
                IntelItem(
                    title=item.title,
                    source_id=feed_id,
                    source_name=feed_name,
                    url=item.url,
                    rank=0,
                    summary=item.summary or "",
                    item_type="rss",
                )
            )
    return rows


def _collect_items(ctx: AppContext) -> List[IntelItem]:
    news_date = _latest_data_date("output/news")
    rss_date = _latest_data_date("output/rss")
    return _flatten_news(ctx, news_date) + _flatten_rss(ctx, rss_date)


def _match_terms(text: str, terms: List[str]) -> List[str]:
    return [t for t in terms if _term_hit(text, t)]


def _is_signal_hit(text: str, any_of: List[str], all_of: List[str]) -> bool:
    if not any(_term_hit(text, term) for term in any_of):
        return False
    return all(_term_hit(text, term) for term in all_of)


def score_item(item: IntelItem, profile: IntelProfile, feedback_state: Dict | None = None) -> IntelResult:
    feedback_state = feedback_state or {}
    text = _norm(f"{item.title} {item.summary} {item.source_name} {item.source_id} {item.url}")

    blocked = {_norm(x) for x in feedback_state.get("blocked_keywords", []) if _norm(x)}
    added = [_norm(x) for x in feedback_state.get("added_keywords", []) if _norm(x)]
    kw_adjust = {k.lower(): float(v) for k, v in feedback_state.get("keyword_adjustments", {}).items()}
    src_adjust = {k.lower(): float(v) for k, v in feedback_state.get("source_adjustments", {}).items()}

    score = 0.0
    matched_keywords: List[str] = []
    matched_buckets: List[str] = []
    matched_sources: List[str] = []
    matched_capital: List[str] = []
    matched_signals: List[str] = []
    detail: Dict = {"bucket_scores": {}, "source_scores": {}, "signal_scores": {}}

    for bucket in profile.keyword_buckets:
        hit_words: List[str] = []
        for keyword in bucket.keywords + added:
            k = _norm(keyword)
            if not k or k in blocked:
                continue
            if _term_hit(text, k):
                hit_words.append(k)

        if not hit_words:
            continue

        bucket_score = 0.0
        for word in hit_words:
            factor = kw_adjust.get(word, 1.0)
            bucket_score += bucket.weight * factor
            matched_keywords.append(word)
        score += bucket_score
        matched_buckets.append(bucket.name)
        detail["bucket_scores"][bucket.name] = round(bucket_score, 3)

    for src in profile.source_watches:
        hits = _match_terms(text, src.match_terms)
        if hits:
            factor = src_adjust.get(_norm(src.name), 1.0)
            src_score = src.weight * factor
            score += src_score
            matched_sources.append(src.name)
            detail["source_scores"][src.name] = round(src_score, 3)

    for cap in profile.capital_watches:
        hits = _match_terms(text, cap.match_terms)
        if hits:
            cap_score = cap.weight
            score += cap_score
            matched_capital.append(cap.name)
            detail["source_scores"][cap.name] = round(cap_score, 3)

    for signal in profile.window_signals:
        if _is_signal_hit(text, signal.any_of, signal.all_of):
            score += signal.weight
            matched_signals.append(signal.name)
            detail["signal_scores"][signal.name] = round(signal.weight, 3)

    has_core_hit = bool(matched_keywords or matched_sources or matched_capital or matched_signals)
    if has_core_hit and item.rank > 0:
        score += max(0.0, (21 - min(item.rank, 20)) * 0.08)

    return IntelResult(
        item=item,
        score=round(score, 4),
        matched_keywords=sorted(set(matched_keywords)),
        matched_buckets=sorted(set(matched_buckets)),
        matched_sources=sorted(set(matched_sources)),
        matched_capital=sorted(set(matched_capital)),
        matched_signals=sorted(set(matched_signals)),
        detail=detail,
    )


def run_intel_pipeline(
    profile_file: str = "config/intelligence.yaml",
    feedback_file: str = "output/intel/feedback_state.json",
    top_k: int = 30,
) -> List[IntelResult]:
    config = load_config()
    ctx = AppContext(config)
    profile = load_intel_profile(profile_file)
    feedback_state = load_feedback_state(feedback_file)
    items = _collect_items(ctx)

    results = [score_item(item, profile, feedback_state) for item in items]
    results = [x for x in results if x.score > 0]
    results.sort(key=lambda x: x.score, reverse=True)
    return results[: max(1, top_k)]


def build_intel_digest(
    results: List[IntelResult],
    output_file: str | None = None,
) -> str:
    lines = [
        "# 行业情报简报",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"情报条目: {len(results)}",
        "",
    ]

    for idx, row in enumerate(results, start=1):
        tags = row.matched_buckets + row.matched_capital + row.matched_signals
        tag_line = " | ".join(tags) if tags else "未命中标签"
        lines.append(f"{idx}. [{row.item.title}]({row.item.url or '#'})")
        lines.append(f"   - 来源: {row.item.source_name} ({row.item.item_type})")
        lines.append(f"   - 评分: {row.score}")
        lines.append(f"   - 标签: {tag_line}")
        if row.matched_keywords:
            lines.append(f"   - 关键词: {', '.join(row.matched_keywords[:10])}")
        lines.append("")

    content = "\n".join(lines)
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return content


def dump_results_json(results: List[IntelResult], output_file: str) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for item in results:
        payload.append(
            {
                "item": asdict(item.item),
                "score": item.score,
                "matched_keywords": item.matched_keywords,
                "matched_buckets": item.matched_buckets,
                "matched_sources": item.matched_sources,
                "matched_capital": item.matched_capital,
                "matched_signals": item.matched_signals,
                "detail": item.detail,
            }
        )
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
