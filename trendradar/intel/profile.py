# coding=utf-8
"""
行业情报配置加载
"""

from pathlib import Path
from typing import Dict, List

import yaml

from .models import IntelProfile, KeywordBucket, SourceWatch, SignalRule


def _to_bucket_list(items: List[Dict]) -> List[KeywordBucket]:
    out = []
    for item in items or []:
        out.append(
            KeywordBucket(
                name=str(item.get("name", "")).strip(),
                weight=float(item.get("weight", 1.0)),
                keywords=[str(x).strip() for x in item.get("keywords", []) if str(x).strip()],
            )
        )
    return [x for x in out if x.name and x.keywords]


def _to_source_list(items: List[Dict]) -> List[SourceWatch]:
    out = []
    for item in items or []:
        terms = [str(x).strip().lower() for x in item.get("match_terms", []) if str(x).strip()]
        out.append(
            SourceWatch(
                name=str(item.get("name", "")).strip(),
                weight=float(item.get("weight", 1.0)),
                match_terms=terms,
            )
        )
    return [x for x in out if x.name and x.match_terms]


def _to_signal_list(items: List[Dict]) -> List[SignalRule]:
    out = []
    for item in items or []:
        any_of = [str(x).strip().lower() for x in item.get("any_of", []) if str(x).strip()]
        all_of = [str(x).strip().lower() for x in item.get("all_of", []) if str(x).strip()]
        out.append(
            SignalRule(
                name=str(item.get("name", "")).strip(),
                weight=float(item.get("weight", 1.0)),
                any_of=any_of,
                all_of=all_of,
            )
        )
    return [x for x in out if x.name and x.any_of]


def load_intel_profile(config_file: str = "config/intelligence.yaml") -> IntelProfile:
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"情报配置文件不存在: {config_file}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    industry = data.get("industry_intel", {})
    return IntelProfile(
        keyword_buckets=_to_bucket_list(industry.get("keyword_buckets", [])),
        source_watches=_to_source_list(industry.get("source_watches", [])),
        capital_watches=_to_source_list(industry.get("capital_watches", [])),
        window_signals=_to_signal_list(industry.get("window_signals", [])),
    )

