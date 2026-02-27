# coding=utf-8
"""
行业情报数据模型
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class KeywordBucket:
    name: str
    weight: float
    keywords: List[str]


@dataclass
class SourceWatch:
    name: str
    weight: float
    match_terms: List[str]


@dataclass
class SignalRule:
    name: str
    weight: float
    any_of: List[str]
    all_of: List[str] = field(default_factory=list)


@dataclass
class IntelProfile:
    keyword_buckets: List[KeywordBucket] = field(default_factory=list)
    source_watches: List[SourceWatch] = field(default_factory=list)
    capital_watches: List[SourceWatch] = field(default_factory=list)
    window_signals: List[SignalRule] = field(default_factory=list)


@dataclass
class IntelItem:
    title: str
    source_id: str
    source_name: str
    url: str
    rank: int = 0
    summary: str = ""
    item_type: str = "hotlist"


@dataclass
class IntelResult:
    item: IntelItem
    score: float
    matched_keywords: List[str] = field(default_factory=list)
    matched_buckets: List[str] = field(default_factory=list)
    matched_sources: List[str] = field(default_factory=list)
    matched_capital: List[str] = field(default_factory=list)
    matched_signals: List[str] = field(default_factory=list)
    detail: Dict = field(default_factory=dict)

