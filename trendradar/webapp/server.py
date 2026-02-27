# coding=utf-8
from __future__ import annotations

import argparse
import re
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import yaml

from trendradar.ai.client import AIClient
from trendradar.intel.feedback import record_feedback, load_feedback_state, save_feedback_state
from trendradar.intel.pipeline import build_intel_digest, dump_results_json, run_intel_pipeline


_CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")
_EN_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-_]{2,20}")
_KEYWORD_STOPWORDS = {
    "我们", "你们", "他们", "这个", "那个", "以及", "但是", "因此", "通过", "进行", "相关", "发布", "公司", "行业", "市场",
    "中国", "美国", "全球", "今日", "昨天", "关注", "新闻", "消息", "报道", "分析", "热点", "趋势", "项目", "产品", "系统",
    "about", "from", "with", "this", "that", "will", "have", "has", "for", "into", "news", "report", "china", "market", "industry",
}
_RECOMMENDED_PLATFORMS = [
    {"id": "toutiao", "name": "今日头条"},
    {"id": "baidu", "name": "百度热搜"},
    {"id": "weibo", "name": "微博"},
    {"id": "zhihu", "name": "知乎"},
    {"id": "douyin", "name": "抖音"},
    {"id": "bilibili-hot-search", "name": "bilibili 热搜"},
    {"id": "wallstreetcn-hot", "name": "华尔街见闻"},
    {"id": "cls-hot", "name": "财联社热门"},
    {"id": "thepaper", "name": "澎湃新闻"},
    {"id": "ifeng", "name": "凤凰网"},
    {"id": "tieba", "name": "贴吧"},
    {"id": "36kr", "name": "36氪"},
    {"id": "huxiu", "name": "虎嗅"},
    {"id": "ithome", "name": "IT之家"},
    {"id": "juejin", "name": "掘金"},
    {"id": "v2ex", "name": "V2EX"},
    {"id": "douban-movie", "name": "豆瓣电影"},
    {"id": "douban-group", "name": "豆瓣小组"},
]
_RECOMMENDED_RSS = [
    {"id": "hacker-news", "name": "Hacker News", "url": "https://hnrss.org/frontpage"},
    {"id": "techcrunch", "name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
    {"id": "theverge", "name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
    {"id": "wired", "name": "Wired", "url": "https://www.wired.com/feed/rss"},
    {"id": "arstechnica", "name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index"},
    {"id": "reuters-world", "name": "Reuters World", "url": "https://feeds.reuters.com/Reuters/worldNews"},
    {"id": "reuters-business", "name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"id": "wsj-world", "name": "WSJ World", "url": "https://feeds.a.dj.com/rss/RSSWorldNews.xml"},
    {"id": "bloomberg-tech", "name": "Bloomberg Technology", "url": "https://feeds.bloomberg.com/technology/news.rss"},
    {"id": "economist", "name": "The Economist", "url": "https://www.economist.com/the-world-this-week/rss.xml"},
    {"id": "ft-world", "name": "Financial Times World", "url": "https://www.ft.com/world?format=rss"},
    {"id": "mit-tech-review", "name": "MIT Technology Review", "url": "https://www.technologyreview.com/feed/"},
    {"id": "infoq-cn", "name": "InfoQ 中文", "url": "https://www.infoq.cn/feed"},
    {"id": "sspai", "name": "少数派", "url": "https://sspai.com/feed"},
    {"id": "ruanyifeng", "name": "阮一峰的网络日志", "url": "http://www.ruanyifeng.com/blog/atom.xml"},
]


def _json_response(h: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Content-Length", str(len(raw)))
    h.end_headers()
    try:
        h.wfile.write(raw)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        # 客户端提前断开（刷新/跳转/网络抖动），无需打印异常堆栈
        pass


def _read_json(h: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(h.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(h.rfile.read(length).decode("utf-8"))


def _j(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _p_intel_json() -> Path:
    return Path("output/intel/latest_digest.json")


def _p_intel_md() -> Path:
    return Path("output/intel/latest_digest.md")


def _p_feedback() -> Path:
    return Path("output/intel/feedback_state.json")


def _p_settings() -> Path:
    return Path("output/webapp/settings.json")


def _p_runtime_cfg() -> Path:
    return Path("output/webapp/runtime_config.yaml")


def _p_base_cfg() -> Path:
    return Path("config/config.yaml")


def _p_timeline() -> Path:
    return Path("config/timeline.yaml")


def _p_intel_cfg() -> Path:
    return Path("config/intelligence.yaml")


def _p_feedback_locks() -> Path:
    return Path("output/webapp/feedback_locks.json")


def _p_task_status() -> Path:
    return Path("output/webapp/task_status.json")


def _p_intel_feed() -> Path:
    return Path("output/intel/feed.json")


def _p_read_state() -> Path:
    return Path("output/webapp/read_state.json")


def _load_base_cfg() -> Dict[str, Any]:
    p = _p_base_cfg()
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _load_intel_cfg() -> Dict[str, Any]:
    p = _p_intel_cfg()
    if not p.exists():
        return {"industry_intel": {"keyword_buckets": []}}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {"industry_intel": {"keyword_buckets": []}}


def _timeline_presets() -> List[str]:
    p = _p_timeline()
    if not p.exists():
        return ["always_on", "morning_evening", "office_hours", "night_owl", "custom"]
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    presets = list((data.get("presets") or {}).keys())
    if "custom" not in presets:
        presets.append("custom")
    return presets or ["always_on", "custom"]


def _defaults(base: Dict[str, Any]) -> Dict[str, Any]:
    channels = ((base.get("notification") or {}).get("channels") or {})
    ai = base.get("ai") or {}
    ai_analysis = base.get("ai_analysis") or {}
    platforms = base.get("platforms") or {}
    rss = base.get("rss") or {}
    return {
        "notification": {
            "enabled": bool((base.get("notification") or {}).get("enabled", True)),
            "wework_webhook_url": ((channels.get("wework") or {}).get("webhook_url", "")),
            "wework_msg_type": ((channels.get("wework") or {}).get("msg_type", "markdown")),
            "feishu_webhook_url": ((channels.get("feishu") or {}).get("webhook_url", "")),
            "dingtalk_webhook_url": ((channels.get("dingtalk") or {}).get("webhook_url", "")),
            "telegram_bot_token": ((channels.get("telegram") or {}).get("bot_token", "")),
            "telegram_chat_id": ((channels.get("telegram") or {}).get("chat_id", "")),
            "email_from": ((channels.get("email") or {}).get("from", "")),
            "email_password": ((channels.get("email") or {}).get("password", "")),
            "email_to": ((channels.get("email") or {}).get("to", "")),
            "email_smtp_server": ((channels.get("email") or {}).get("smtp_server", "")),
            "email_smtp_port": ((channels.get("email") or {}).get("smtp_port", "")),
            "slack_webhook_url": ((channels.get("slack") or {}).get("webhook_url", "")),
            "generic_webhook_url": ((channels.get("generic_webhook") or {}).get("webhook_url", "")),
        },
        "schedule": {
            "enabled": bool(((base.get("schedule") or {}).get("enabled", True))),
            "preset": str(((base.get("schedule") or {}).get("preset", "morning_evening"))),
        },
        "sources": {
            "platforms_enabled": bool(platforms.get("enabled", True)),
            "platforms": platforms.get("sources", []) or [],
            "rss_enabled": bool(rss.get("enabled", True)),
            "rss_feeds": rss.get("feeds", []) or [],
        },
        "llm": {
            "model": str(ai.get("model", "")),
            "api_key": str(ai.get("api_key", "")),
            "api_base": str(ai.get("api_base", "")),
            "timeout": int(ai.get("timeout", 120) or 120),
            "temperature": float(ai.get("temperature", 1.0) or 1.0),
            "max_tokens": int(ai.get("max_tokens", 5000) or 5000),
            "analysis_enabled": bool(ai_analysis.get("enabled", False)),
            "analysis_language": str(ai_analysis.get("language", "Chinese")),
            "analysis_mode": str(ai_analysis.get("mode", "follow_report")),
        },
    }


def _load_settings() -> Dict[str, Any]:
    d = _defaults(_load_base_cfg())
    saved = _j(_p_settings()) or {}
    d["notification"].update(saved.get("notification", {}))
    d["schedule"].update(saved.get("schedule", {}))
    d["sources"].update(saved.get("sources", {}))
    d["llm"].update(saved.get("llm", {}))
    return d


def _save_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    cur = _load_settings()
    if "notification" in payload:
        cur["notification"].update(payload.get("notification", {}))
    if "schedule" in payload:
        cur["schedule"].update(payload.get("schedule", {}))
    if "sources" in payload:
        cur["sources"].update(payload.get("sources", {}))
    if "llm" in payload:
        cur["llm"].update(payload.get("llm", {}))
    if cur["schedule"].get("preset") not in _timeline_presets():
        raise ValueError("无效 preset")
    p = _p_settings()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    return cur


def _make_ai_client_from_settings(settings: Dict[str, Any]) -> AIClient | None:
    llm = settings.get("llm", {}) or {}
    model = str(llm.get("model", "")).strip()
    api_key = str(llm.get("api_key", "")).strip()
    if not model or not api_key:
        return None
    config = {
        "MODEL": model,
        "API_KEY": api_key,
        "API_BASE": str(llm.get("api_base", "")).strip(),
        "TIMEOUT": int(llm.get("timeout", 120) or 120),
        "TEMPERATURE": float(llm.get("temperature", 0.2) or 0.2),
        "MAX_TOKENS": int(llm.get("max_tokens", 1200) or 1200),
    }
    return AIClient(config)


def _safe_json_loads(raw_text: str) -> Any:
    text = (raw_text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # 兼容模型返回 markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            return None
    return None


def _source_merge(base: List[Dict[str, str]], incoming: List[Dict[str, str]], is_rss: bool = False) -> Tuple[List[Dict[str, str]], int]:
    merged = list(base or [])
    exists = {str(x.get("id", "")).strip().lower() for x in merged if str(x.get("id", "")).strip()}
    added = 0
    for item in incoming:
        sid = str(item.get("id", "")).strip()
        if not sid or sid.lower() in exists:
            continue
        name = str(item.get("name", sid)).strip() or sid
        row: Dict[str, str] = {"id": sid, "name": name}
        if is_rss:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            row["url"] = url
        merged.append(row)
        exists.add(sid.lower())
        added += 1
    return merged, added


def _suggest_keywords_from_feedback(limit: int = 20) -> Dict[str, Any]:
    feedback = load_feedback_state(str(_p_feedback()))
    events = feedback.get("events", []) or []
    accepted = [ev for ev in events if int(ev.get("rating", 0) or 0) >= 4]
    if not accepted:
        return {"ok": True, "items": [], "accepted_count": 0, "used_llm": False}

    existing_keywords = set()
    for bucket in _read_keyword_buckets():
        for kw in bucket.get("keywords", []):
            k = str(kw).strip().lower()
            if k:
                existing_keywords.add(k)
    for kw in feedback.get("added_keywords", []) or []:
        k = str(kw).strip().lower()
        if k:
            existing_keywords.add(k)
    for kw in feedback.get("blocked_keywords", []) or []:
        k = str(kw).strip().lower()
        if k:
            existing_keywords.add(k)

    counter: Counter = Counter()
    freq_counter: Counter = Counter()
    for ev in accepted:
        weight = 1.0 + max(0, int(ev.get("rating", 4)) - 4) * 0.4
        title = str(ev.get("item_title", "") or "")
        note = str(ev.get("note", "") or "")
        matched = [str(x).strip() for x in (ev.get("keywords", []) or []) if str(x).strip()]
        text = " ".join([title, note, " ".join(matched)]).lower()
        if not text.strip():
            continue
        for token in _CJK_TOKEN_RE.findall(text) + _EN_TOKEN_RE.findall(text):
            t = token.strip().lower()
            if len(t) < 2 or t in _KEYWORD_STOPWORDS:
                continue
            if t in existing_keywords:
                continue
            counter[t] += weight
            freq_counter[t] += 1

    raw_rule_items = []
    for kw, score in counter.most_common(max(limit * 3, 60)):
        if freq_counter[kw] < 2 and score < 2.2:
            continue
        raw_rule_items.append({"keyword": kw, "score": round(float(score), 3), "source": "rule", "reason": "高分反馈标题共现"})
    top_titles = [str(ev.get("item_title", "")).strip() for ev in accepted if str(ev.get("item_title", "")).strip()][:20]

    llm_items: List[Dict[str, Any]] = []
    used_llm = False
    ai_client = _make_ai_client_from_settings(_load_settings())
    if ai_client and top_titles:
        try:
            seed_words = [x["keyword"] for x in raw_rule_items[:30]]
            messages = [
                {"role": "system", "content": "你是关键词工程助手。请只输出 JSON 数组，不要解释。"},
                {
                    "role": "user",
                    "content": (
                        "下面是用户已判定为有价值的新闻标题，请提炼可用于持续监控的关键词（2-8字优先，避免通用词）。"
                        "返回格式: [{\"keyword\":\"...\",\"reason\":\"...\"}]，最多 20 条。\n\n"
                        f"标题样本:\n- " + "\n- ".join(top_titles) + "\n\n"
                        f"规则候选参考: {', '.join(seed_words)}"
                    ),
                },
            ]
            resp = ai_client.chat(messages, temperature=0.1, max_tokens=900)
            parsed = _safe_json_loads(resp)
            if isinstance(parsed, list):
                for row in parsed[:12]:
                    if not isinstance(row, dict):
                        continue
                    kw = str(row.get("keyword", "")).strip().lower()
                    if not kw or len(kw) < 2 or kw in existing_keywords or kw in _KEYWORD_STOPWORDS:
                        continue
                    llm_items.append(
                        {
                            "keyword": kw,
                            "score": 999.0,  # LLM 候选优先展示
                            "source": "llm",
                            "reason": str(row.get("reason", "LLM 提炼")).strip() or "LLM 提炼",
                        }
                    )
                used_llm = True
        except Exception:
            used_llm = False

    merged: List[Dict[str, Any]] = []
    seen = set()
    for row in llm_items + raw_rule_items:
        kw = str(row.get("keyword", "")).strip().lower()
        if not kw or kw in seen:
            continue
        seen.add(kw)
        merged.append(row)
        if len(merged) >= max(1, limit):
            break

    return {"ok": True, "items": merged, "accepted_count": len(accepted), "used_llm": used_llm}


def _build_intel_summary(results: List[Any], top_k: int) -> Dict[str, Any]:
    top_sources: Counter = Counter()
    top_buckets: Counter = Counter()
    top_keywords: Counter = Counter()
    top_signals: Counter = Counter()
    for row in results:
        item = getattr(row, "item", None)
        source_name = str(getattr(item, "source_name", "") or "").strip()
        if source_name:
            top_sources[source_name] += 1
        for b in getattr(row, "matched_buckets", []) or []:
            top_buckets[str(b)] += 1
        for k in getattr(row, "matched_keywords", []) or []:
            top_keywords[str(k)] += 1
        for s in getattr(row, "matched_signals", []) or []:
            top_signals[str(s)] += 1
    return {
        "requested_top_k": top_k,
        "hit_count": len(results),
        "top_sources": [{"name": k, "count": v} for k, v in top_sources.most_common(5)],
        "top_buckets": [{"name": k, "count": v} for k, v in top_buckets.most_common(5)],
        "top_keywords": [{"name": k, "count": v} for k, v in top_keywords.most_common(8)],
        "top_signals": [{"name": k, "count": v} for k, v in top_signals.most_common(5)],
    }


def _build_ai_intel_summary(results: List[Any], summary: Dict[str, Any]) -> str:
    ai_client = _make_ai_client_from_settings(_load_settings())
    if not ai_client:
        return ""
    sample_titles = []
    for row in results[:20]:
        item = getattr(row, "item", None)
        title = str(getattr(item, "title", "") or "").strip()
        if title:
            sample_titles.append(title)
    if not sample_titles:
        return ""
    payload = {
        "hit_count": summary.get("hit_count", 0),
        "top_sources": summary.get("top_sources", []),
        "top_buckets": summary.get("top_buckets", []),
        "top_keywords": summary.get("top_keywords", []),
        "top_signals": summary.get("top_signals", []),
        "sample_titles": sample_titles[:12],
    }
    try:
        messages = [
            {"role": "system", "content": "你是行业情报运营助手。请用中文给出2-3句执行解读，简洁具体。"},
            {"role": "user", "content": f"请基于以下 JSON 生成执行总结：\n{json.dumps(payload, ensure_ascii=False)}"},
        ]
        return (ai_client.chat(messages, temperature=0.2, max_tokens=220) or "").strip()
    except Exception:
        return ""

def _build_runtime_cfg(settings: Dict[str, Any]) -> Path:
    src = _p_base_cfg()
    if not src.exists():
        raise FileNotFoundError("config/config.yaml 不存在")
    base = yaml.safe_load(src.read_text(encoding="utf-8")) or {}

    n = settings.get("notification", {})
    notif = base.setdefault("notification", {})
    notif["enabled"] = bool(n.get("enabled", True))
    channels = notif.setdefault("channels", {})
    channels.setdefault("wework", {})
    channels["wework"]["webhook_url"] = str(n.get("wework_webhook_url", "")).strip()
    channels["wework"]["msg_type"] = str(n.get("wework_msg_type", "markdown")).strip() or "markdown"
    channels.setdefault("feishu", {})
    channels["feishu"]["webhook_url"] = str(n.get("feishu_webhook_url", "")).strip()
    channels.setdefault("dingtalk", {})
    channels["dingtalk"]["webhook_url"] = str(n.get("dingtalk_webhook_url", "")).strip()
    channels.setdefault("telegram", {})
    channels["telegram"]["bot_token"] = str(n.get("telegram_bot_token", "")).strip()
    channels["telegram"]["chat_id"] = str(n.get("telegram_chat_id", "")).strip()
    channels.setdefault("email", {})
    channels["email"]["from"] = str(n.get("email_from", "")).strip()
    channels["email"]["password"] = str(n.get("email_password", "")).strip()
    channels["email"]["to"] = str(n.get("email_to", "")).strip()
    channels["email"]["smtp_server"] = str(n.get("email_smtp_server", "")).strip()
    channels["email"]["smtp_port"] = str(n.get("email_smtp_port", "")).strip()
    channels.setdefault("slack", {})
    channels["slack"]["webhook_url"] = str(n.get("slack_webhook_url", "")).strip()
    channels.setdefault("generic_webhook", {})
    channels["generic_webhook"]["webhook_url"] = str(n.get("generic_webhook_url", "")).strip()

    s = settings.get("schedule", {})
    sch = base.setdefault("schedule", {})
    sch["enabled"] = bool(s.get("enabled", True))
    sch["preset"] = str(s.get("preset", "morning_evening")).strip() or "morning_evening"

    src_cfg = settings.get("sources", {})
    platforms = base.setdefault("platforms", {})
    platforms["enabled"] = bool(src_cfg.get("platforms_enabled", True))
    platforms["sources"] = src_cfg.get("platforms", []) or []
    rss = base.setdefault("rss", {})
    rss["enabled"] = bool(src_cfg.get("rss_enabled", True))
    rss["feeds"] = src_cfg.get("rss_feeds", []) or []

    llm = settings.get("llm", {})
    ai = base.setdefault("ai", {})
    ai["model"] = str(llm.get("model", "")).strip()
    ai["api_key"] = str(llm.get("api_key", "")).strip()
    ai["api_base"] = str(llm.get("api_base", "")).strip()
    ai["timeout"] = int(llm.get("timeout", 120) or 120)
    ai["temperature"] = float(llm.get("temperature", 1.0) or 1.0)
    ai["max_tokens"] = int(llm.get("max_tokens", 5000) or 5000)

    ai_analysis = base.setdefault("ai_analysis", {})
    ai_analysis["enabled"] = bool(llm.get("analysis_enabled", False))
    ai_analysis["language"] = str(llm.get("analysis_language", "Chinese")).strip() or "Chinese"
    ai_analysis["mode"] = str(llm.get("analysis_mode", "follow_report")).strip() or "follow_report"

    out = _p_runtime_cfg()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # 调度系统会按 config_path 同目录查找 timeline.yaml，这里同步一份到 runtime 目录
    src_timeline = _p_timeline()
    dst_timeline = out.parent / "timeline.yaml"
    if src_timeline.exists():
        shutil.copyfile(src_timeline, dst_timeline)

    return out


def _run_crawler(timeout_sec: int = 300) -> Tuple[bool, str]:
    start = datetime.now()
    task = _load_task_status()
    task["crawler"]["last_start"] = start.isoformat(timespec="seconds")
    task["crawler"]["running"] = True
    _save_task_status(task)
    try:
        cfg = _build_runtime_cfg(_load_settings())
        env = dict(os.environ)
        env["CONFIG_PATH"] = str(cfg)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["DOCKER_CONTAINER"] = "true"
        r = subprocess.run(
            [sys.executable, "-m", "trendradar"],
            capture_output=True,
            text=False,
            timeout=timeout_sec,
            check=False,
            env=env,
        )

        def _decode(raw: bytes) -> str:
            if not raw:
                return ""
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("gbk", errors="replace")

        stdout = _decode(r.stdout or b"")
        stderr = _decode(r.stderr or b"")
        out = stdout + ("\n" + stderr if stderr else "")
        end = datetime.now()
        ok = r.returncode == 0
        task = _load_task_status()
        task["crawler"].update(
            {
                "last_end": end.isoformat(timespec="seconds"),
                "ok": ok,
                "running": False,
                "duration_sec": round((end - start).total_seconds(), 2),
                "message": out.strip()[-500:] if out else "",
            }
        )
        _save_task_status(task)

        # 抓取成功后自动刷新一次情报，避免前端出现“抓取成功但情报为空”的状态错位
        if ok:
            try:
                intel = _run_intel(top_k=30)
                out = (out.strip() + f"\n[webapp] 情报已刷新，命中 {intel.get('count', 0)} 条").strip()
            except Exception as intel_err:
                out = (out.strip() + f"\n[webapp] 情报刷新失败: {intel_err}").strip()
        return ok, out.strip()
    except Exception as e:
        end = datetime.now()
        task = _load_task_status()
        task["crawler"].update(
            {
                "last_end": end.isoformat(timespec="seconds"),
                "ok": False,
                "running": False,
                "duration_sec": round((end - start).total_seconds(), 2),
                "message": f"执行失败: {e}",
            }
        )
        _save_task_status(task)
        return False, f"执行失败: {e}"


def _run_intel(top_k: int = 30) -> Dict[str, Any]:
    start = datetime.now()
    task = _load_task_status()
    task["intel"]["last_start"] = start.isoformat(timespec="seconds")
    task["intel"]["running"] = True
    _save_task_status(task)
    md = _p_intel_md()
    js = _p_intel_json()
    md.parent.mkdir(parents=True, exist_ok=True)
    try:
        results = run_intel_pipeline(profile_file="config/intelligence.yaml", feedback_file=str(_p_feedback()), top_k=top_k)
        build_intel_digest(results, output_file=str(md))
        dump_results_json(results, str(js))
        _merge_into_intel_feed(_j(js) or [])
        summary = _build_intel_summary(results, top_k)
        ai_summary = _build_ai_intel_summary(results, summary)
        top_sources_text = ", ".join([f"{x['name']}({x['count']})" for x in summary.get("top_sources", [])[:3]]) or "-"
        top_keywords_text = ", ".join([f"{x['name']}({x['count']})" for x in summary.get("top_keywords", [])[:5]]) or "-"
        message_parts = [
            f"完成: 命中 {len(results)} 条",
            f"TopK={top_k}",
            f"来源Top: {top_sources_text}",
            f"关键词Top: {top_keywords_text}",
            f"输出: {str(md).replace('\\\\', '/')}",
        ]
        if ai_summary:
            message_parts.append(f"AI: {ai_summary}")
        final_message = " | ".join(message_parts)
        end = datetime.now()
        task = _load_task_status()
        task["intel"].update(
            {
                "last_end": end.isoformat(timespec="seconds"),
                "ok": True,
                "running": False,
                "duration_sec": round((end - start).total_seconds(), 2),
                "count": len(results),
                "message": final_message,
            }
        )
        _save_task_status(task)
        return {
            "count": len(results),
            "md_file": str(md).replace("\\", "/"),
            "json_file": str(js).replace("\\", "/"),
            "summary": summary,
            "ai_summary": ai_summary,
        }
    except Exception as e:
        end = datetime.now()
        task = _load_task_status()
        task["intel"].update(
            {
                "last_end": end.isoformat(timespec="seconds"),
                "ok": False,
                "running": False,
                "duration_sec": round((end - start).total_seconds(), 2),
                "count": 0,
                "message": str(e),
            }
        )
        _save_task_status(task)
        raise


def _recent_reports(n: int = 20) -> List[str]:
    root = Path("output/html")
    if not root.exists():
        return []
    fs = sorted(root.rglob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(x).replace("\\", "/") for x in fs[:n]]


def _has_notify(n: Dict[str, Any]) -> bool:
    return bool(
        n.get("wework_webhook_url")
        or n.get("feishu_webhook_url")
        or n.get("dingtalk_webhook_url")
        or (n.get("telegram_bot_token") and n.get("telegram_chat_id"))
        or (n.get("email_from") and n.get("email_password") and n.get("email_to"))
        or n.get("slack_webhook_url")
        or n.get("generic_webhook_url")
    )


def _read_keyword_buckets() -> List[Dict[str, Any]]:
    data = _load_intel_cfg()
    intel = data.get("industry_intel") or {}
    buckets = intel.get("keyword_buckets") or []
    cleaned: List[Dict[str, Any]] = []
    for idx, b in enumerate(buckets, start=1):
        cleaned.append(
            {
                "name": str(b.get("name", f"分组{idx}")).strip() or f"分组{idx}",
                "flow_stage": str(b.get("flow_stage", "观察")).strip() or "观察",
                "weight": float(b.get("weight", 1.0) or 1.0),
                "keywords": [str(x).strip() for x in (b.get("keywords") or []) if str(x).strip()],
            }
        )
    return cleaned


def _save_keyword_buckets(buckets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    path = _p_intel_cfg()
    data = _load_intel_cfg()
    intel = data.setdefault("industry_intel", {})
    normalized = []
    for idx, b in enumerate(buckets, start=1):
        keywords = [str(x).strip() for x in (b.get("keywords") or []) if str(x).strip()]
        if not keywords:
            continue
        normalized.append(
            {
                "name": str(b.get("name", f"分组{idx}")).strip() or f"分组{idx}",
                "flow_stage": str(b.get("flow_stage", "观察")).strip() or "观察",
                "weight": float(b.get("weight", 1.0) or 1.0),
                "keywords": keywords,
            }
        )
    intel["keyword_buckets"] = normalized
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return normalized


def _feedback_key(title: str, source: str, url: str) -> str:
    raw = f"{(source or '').strip().lower()}|{(title or '').strip().lower()}|{(url or '').strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _feedback_raw_key(title: str, source: str, url: str) -> str:
    return f"{(source or '').strip().lower()}|{(title or '').strip().lower()}|{(url or '').strip().lower()}"


def _remove_feedback_lock(item_title: str, item_source: str, item_url: str) -> bool:
    if not item_title:
        return False
    locks = _load_feedback_locks()
    items = locks.get("items", {})
    target_key = _feedback_key(item_title, item_source, item_url)
    removed = items.pop(target_key, None)
    if not removed:
        # 兼容历史 key 差异，按原始字段匹配
        raw = _feedback_raw_key(item_title, item_source, item_url)
        for k, meta in list(items.items()):
            if _feedback_raw_key(meta.get("title", ""), meta.get("source", ""), meta.get("url", "")) == raw:
                removed = items.pop(k)
                break
    _save_feedback_locks(locks)
    return bool(removed)


def _load_feedback_locks() -> Dict[str, Any]:
    return _j(_p_feedback_locks()) or {"items": {}}


def _save_feedback_locks(data: Dict[str, Any]) -> None:
    p = _p_feedback_locks()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _feedback_locks_raw_map(locks: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for _, meta in (locks.get("items") or {}).items():
        k = _feedback_raw_key(meta.get("title", ""), meta.get("source", ""), meta.get("url", ""))
        out[k] = meta
    return out


def _load_task_status() -> Dict[str, Any]:
    default = {
        "crawler": {"last_start": "", "last_end": "", "ok": None, "running": False, "duration_sec": 0, "message": ""},
        "intel": {"last_start": "", "last_end": "", "ok": None, "running": False, "duration_sec": 0, "count": 0, "message": ""},
    }
    data = _j(_p_task_status()) or {}
    out = {"crawler": dict(default["crawler"]), "intel": dict(default["intel"])}
    out["crawler"].update(data.get("crawler", {}) if isinstance(data.get("crawler", {}), dict) else {})
    out["intel"].update(data.get("intel", {}) if isinstance(data.get("intel", {}), dict) else {})
    return out


def _save_task_status(data: Dict[str, Any]) -> None:
    p = _p_task_status()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_read_state() -> Dict[str, Any]:
    return _j(_p_read_state()) or {"items": {}}


def _save_read_state(data: Dict[str, Any]) -> None:
    p = _p_read_state()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _intel_row_key(row: Dict[str, Any]) -> str:
    item = row.get("item") or {}
    source = str(item.get("source_name") or item.get("source_id") or "").strip().lower()
    title = str(item.get("title") or "").strip().lower()
    url = str(item.get("url") or "").strip().lower()
    return _feedback_raw_key(title, source, url)


def _load_intel_feed() -> Dict[str, Any]:
    return _j(_p_intel_feed()) or {"items": []}


def _save_intel_feed(data: Dict[str, Any]) -> None:
    p = _p_intel_feed()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_into_intel_feed(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    current = _load_intel_feed()
    items = current.get("items", [])
    by_key: Dict[str, Dict[str, Any]] = {}
    for it in items:
        k = str(it.get("key", "")).strip()
        if k:
            by_key[k] = it

    for row in rows:
        key = _intel_row_key(row)
        if not key:
            continue
        if key in by_key:
            by_key[key]["row"] = row
            by_key[key]["last_seen"] = now
            by_key[key]["seen_count"] = int(by_key[key].get("seen_count", 1)) + 1
        else:
            by_key[key] = {"key": key, "first_seen": now, "last_seen": now, "seen_count": 1, "row": row}

    merged = list(by_key.values())
    merged.sort(key=lambda x: x.get("last_seen", ""), reverse=True)
    merged = merged[:500]
    out = {"items": merged}
    _save_intel_feed(out)
    return out


def _intel_feed_with_read() -> Dict[str, Any]:
    feed = _load_intel_feed()
    if not feed.get("items"):
        latest = _j(_p_intel_json()) or []
        if latest:
            feed = _merge_into_intel_feed(latest)
    read = _load_read_state().get("items", {})
    enriched = []
    for item in feed.get("items", []):
        key = item.get("key", "")
        row = item.get("row", {})
        enriched.append(
            {
                **item,
                "read": key in read,
                "read_at": read.get(key, ""),
                "row": row,
            }
        )
    return {"items": enriched}


def _intel_unread_count() -> int:
    feed = _intel_feed_with_read()
    return sum(1 for item in (feed.get("items") or []) if not bool(item.get("read")))


def _latest_db_file(folder: str) -> Path | None:
    root = Path(folder)
    if not root.exists():
        return None
    files = sorted(root.glob("*.db"), reverse=True)
    return files[0] if files else None


def _discover_sources() -> Dict[str, List[Dict[str, str]]]:
    discovered_platforms: List[Dict[str, str]] = []
    discovered_rss: List[Dict[str, str]] = []

    news_db = _latest_db_file("output/news")
    if news_db:
        try:
            conn = sqlite3.connect(str(news_db))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, name FROM platforms ORDER BY id").fetchall()
            discovered_platforms = [{"id": str(r["id"]), "name": str(r["name"])} for r in rows]
            conn.close()
        except Exception:
            pass

    rss_db = _latest_db_file("output/rss")
    if rss_db:
        try:
            conn = sqlite3.connect(str(rss_db))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, name, feed_url FROM rss_feeds ORDER BY id").fetchall()
            discovered_rss = [
                {"id": str(r["id"]), "name": str(r["name"]), "url": str(r["feed_url"] or "")}
                for r in rows
            ]
            conn.close()
        except Exception:
            pass

    return {"platforms": discovered_platforms, "rss_feeds": discovered_rss}


def _recommended_sources() -> Dict[str, List[Dict[str, str]]]:
    return {"platforms": list(_RECOMMENDED_PLATFORMS), "rss_feeds": list(_RECOMMENDED_RSS)}


def _bootstrap_sources_into_settings() -> Dict[str, Any]:
    settings = _load_settings()
    sources = settings.setdefault("sources", {})
    platforms = sources.get("platforms", []) or []
    rss_feeds = sources.get("rss_feeds", []) or []
    platforms, added_p = _source_merge(platforms, _RECOMMENDED_PLATFORMS, is_rss=False)
    rss_feeds, added_f = _source_merge(rss_feeds, _RECOMMENDED_RSS, is_rss=True)
    sources["platforms"] = platforms
    sources["rss_feeds"] = rss_feeds
    saved = _save_settings({"sources": sources})
    return {
        "ok": True,
        "sources": saved.get("sources", {}),
        "added_platforms": added_p,
        "added_rss": added_f,
        "recommended_platforms": len(_RECOMMENDED_PLATFORMS),
        "recommended_rss": len(_RECOMMENDED_RSS),
    }


def _html(title: str, active: str, body: str, script: str) -> str:
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{title}</title><style>
body{{margin:0;font-family:Segoe UI,Microsoft YaHei,sans-serif;background:radial-gradient(circle at 0 0,#def7ec 0,transparent 20%),radial-gradient(circle at 100% 0,#dbeafe 0,transparent 25%),#f5f7fb;color:#111827}}
.top{{padding:14px 18px;background:linear-gradient(120deg,#0f766e,#1d4ed8);color:#fff;box-shadow:0 8px 20px rgba(15,118,110,.18)}}
.title-main{{font-size:20px;font-weight:800;letter-spacing:.2px}}
.sub-main{{font-size:12px;opacity:.92;margin-top:4px}}
.nav{{padding:10px 16px;background:#fff;border-bottom:1px solid #dbe4ee;display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;z-index:5}}
.nav a{{text-decoration:none;color:#0f172a;padding:7px 11px;border-radius:9px;border:1px solid transparent;font-size:13px;transition:all .18s ease}}
.nav a:hover{{background:#f8fafc;border-color:#dbe4ee}}
.nav a.on{{background:#eff6ff;border-color:#93c5fd;font-weight:700;box-shadow:0 1px 0 #bfdbfe inset}}
.badge-red{{display:none;align-items:center;justify-content:center;min-width:18px;height:18px;padding:0 5px;border-radius:999px;background:#dc2626;color:#fff;font-size:11px;font-weight:700;margin-left:6px}}
.wrap{{max-width:1120px;margin:14px auto;padding:0 12px 24px}}
.card{{background:#fff;border:1px solid #dbe4ee;border-radius:12px;padding:14px;margin-bottom:12px;box-shadow:0 3px 10px rgba(15,23,42,.04)}}
.t{{font-weight:800;margin-bottom:8px;font-size:15px}} .hint{{font-size:12px;color:#4b5563;margin-bottom:10px;line-height:1.5}}
.row{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;align-items:flex-start}} .col{{flex:1;min-width:240px}}
input,select,textarea{{width:100%;border:1px solid #dbe4ee;border-radius:9px;padding:9px;font-size:13px;background:#fff}} textarea{{min-height:78px}}
input:focus,select:focus,textarea:focus{{outline:none;border-color:#60a5fa;box-shadow:0 0 0 3px rgba(96,165,250,.2)}}
button{{border:1px solid #cbd5e1;border-radius:9px;padding:8px 12px;color:#0f172a;background:#fff;font-weight:600;cursor:pointer;transition:transform .08s ease,opacity .16s ease,background .16s ease}}
button:hover{{opacity:.95;background:#f8fafc}} button:active{{transform:translateY(1px)}}
button:disabled{{opacity:.55;cursor:not-allowed;background:#f3f4f6;color:#6b7280;border-color:#e5e7eb}}
.alt{{background:#eff6ff;border-color:#bfdbfe;color:#1e3a8a}}
.good{{background:#ecfdf5;border-color:#86efac;color:#166534}}
.mid{{background:#f8fafc;border-color:#cbd5e1;color:#334155}}
.bad{{background:#fef2f2;border-color:#fecaca;color:#991b1b}}
.primary{{background:#0f766e;border-color:#0f766e;color:#fff}}
.soft{{font-size:12px;padding:5px 10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:8px 6px;border-bottom:1px solid #eef2f7;vertical-align:top}}
th{{background:#f8fafc;font-size:12px;color:#334155}}
.log{{white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:9px;max-height:240px;overflow:auto;font-size:12px}}
.meta{{font-size:12px;color:#4b5563;line-height:1.45}} .pill{{display:inline-block;padding:2px 8px;border-radius:999px;background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8;font-size:11px;font-weight:700}}
a.lk{{color:#0c4a6e;text-decoration:none}} a.lk:hover{{text-decoration:underline}}
.prog{{height:10px;background:#e5e7eb;border-radius:999px;overflow:hidden}}
.prog > i{{display:block;height:100%;width:0%;background:#3b82f6;transition:width .35s ease,background .25s ease}}
.prog-meta{{font-size:12px;color:#475569;margin-top:4px}}
</style></head><body>
<div class='top'><div class='title-main'>TrendRadar 本地控制台</div><div class='sub-main'>按流程使用：先配置 → 再执行 → 看结果 → 做反馈</div></div>
<div class='nav'>
<a class='{ 'on' if active=='sources' else '' }' href='/sources'>1 数据源</a>
<a class='{ 'on' if active=='keywords' else '' }' href='/keywords'>2 关键词</a>
<a class='{ 'on' if active=='llm' else '' }' href='/llm'>3 大模型</a>
<a class='{ 'on' if active=='notification' else '' }' href='/notification'>4 通知</a>
<a class='{ 'on' if active=='schedule' else '' }' href='/schedule'>5 调度</a>
<a class='{ 'on' if active=='dashboard' else '' }' href='/dashboard'>6 运行任务</a>
<a class='{ 'on' if active=='intel' else '' }' href='/intel'>7 看结果<span id='navIntelBadge' class='badge-red'></span></a>
<a class='{ 'on' if active=='feedback' else '' }' href='/feedback'>8 用户反馈</a>
</div><div class='wrap'>{body}</div>
<script>
let __pendingLockedApi = 0;
function __setButtonsBusy(b){{document.querySelectorAll('button').forEach(btn=>{{if(btn.dataset.noBusy==='true')return;btn.disabled=b;}});}}
async function api(path,method='GET',body=null,lock=true){{
  if(lock){{if(__pendingLockedApi===0)__setButtonsBusy(true);__pendingLockedApi+=1;}}
  try {{
    const r=await fetch(path,{{method,headers:{{'Content-Type':'application/json'}},body:body?JSON.stringify(body):null}});
    return await r.json();
  }} finally {{
    if(lock){{__pendingLockedApi-=1;if(__pendingLockedApi<=0){{__pendingLockedApi=0;__setButtonsBusy(false);}}}}
  }}
}}
function esc(s){{return String(s||'').replace(/[&<>\"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}}[m]));}}
async function refreshIntelBadge(){{
  try {{
    const d = await api('/api/status','GET',null,false);
    const n = Number(d.intel_unread||0) || 0;
    const el = document.getElementById('navIntelBadge');
    if(!el) return;
    el.textContent = n>99 ? '99+' : String(n);
    el.style.display = n>0 ? 'inline-flex' : 'none';
  }} catch (_err) {{}}
}}
refreshIntelBadge();
setInterval(refreshIntelBadge, 30000);
{script}
</script></body></html>"""


def _page_dashboard() -> str:
    body = """
<div class='card'><div class='t'>运行任务</div>
<div class='hint'>第 6 步：点击按钮执行任务。建议先完成 1-5 步配置。</div>
<div class='row'><button class='primary' onclick='runCrawler()'>开始抓取（含自动刷新情报）</button><button class='alt' onclick='runIntel()'>单独运行情报分析</button></div>
<div class='row'><div class='col'><div class='meta'>情报 TopK</div><input id='topk' value='30'></div></div>
<div class='hint'>抓取会自动使用你在“通知/调度/模型/数据源/关键词”页保存的配置（运行时注入，不改原始 config/config.yaml）。</div>
<div id='log' class='log'>任务日志将在这里显示。点击上方按钮后会实时追加关键过程。</div></div>
<div class='card'><div class='t'>任务状态中心 <span class='pill'>实时</span></div>
<div id='s1' class='meta'>正在读取系统状态...</div>
<div id='s2' class='meta'></div>
<div class='prog'><i id='crawlerBar'></i></div><div id='crawlerBarText' class='prog-meta'>抓取任务进度：待执行</div>
<div id='s3' class='meta'></div>
<div class='prog'><i id='intelBar'></i></div><div id='intelBarText' class='prog-meta'>情报任务进度：待执行</div>
<div id='s4' class='meta'></div>
</div>"""
    script = """
function w(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent=`[${t}] ${m}\\n`+e.textContent;}
function presetZh(p){const m={always_on:'全天候',morning_evening:'早晚节奏',office_hours:'工作时段',night_owl:'夜间复盘',custom:'自定义'};return m[p]||p||'-';}
function taskLine(name,t){if(!t)return `${name}: 暂无执行记录`;const st=t.running?'执行中':(t.ok===true?'成功':(t.ok===false?'失败':'未执行'));const msg=(t.message||'').replace(/\\s+/g,' ').trim();return `${name}: ${st} | 开始:${t.last_start||'-'} | 结束:${t.last_end||'-'} | 耗时:${t.duration_sec||0}s${msg?` | 摘要:${msg.slice(-120)}`:''}`;}
function extractHighlights(output){const lines=String(output||'').split('\\n').map(x=>x.trim()).filter(Boolean);const picks=lines.filter(x=>x.includes('处理完成')||x.includes('[RSS]')||x.includes('[调度]')||x.includes('[推送]')||x.includes('[AI]')||x.includes('[webapp]')||x.includes('HTML报告已生成')||x.includes('新增'));return (picks.length?picks:lines).slice(-10);}
function ts(v){const n=Date.parse(v||'');return Number.isNaN(n)?0:n;}
function setBar(id,textId,pct,label,color){const bar=document.getElementById(id);const text=document.getElementById(textId);if(bar)bar.style.width=`${Math.max(0,Math.min(100,pct))}%`;if(bar)bar.style.background=color||'#3b82f6';if(text)text.textContent=label;}
function renderProgress(task,id,textId,kind){if(!task){setBar(id,textId,0,`${kind}进度：待执行`,'#94a3b8');return;}if(task.running){const elapsed=Math.max(0,(Date.now()-ts(task.last_start))/1000);const expected=kind==='抓取任务'?180:60;const pct=Math.min(92,8+(elapsed/expected)*84);setBar(id,textId,pct,`${kind}进度：执行中 ${Math.floor(elapsed)}s`,'#2563eb');return;}if(task.ok===true){setBar(id,textId,100,`${kind}进度：已完成（${task.duration_sec||0}s）`,'#16a34a');return;}if(task.ok===false){setBar(id,textId,100,`${kind}进度：失败（${task.duration_sec||0}s）`,'#dc2626');return;}setBar(id,textId,0,`${kind}进度：待执行`,'#94a3b8');}
async function status(){const d=await api('/api/status');document.getElementById('s1').textContent=`系统概览：情报 ${d.intel_count} 条 | 已反馈 ${d.feedback_events} 条 | 通知 ${d.notification_configured?'已配置':'未配置'} | 调度 ${d.schedule_enabled?'开启':'关闭'}（${presetZh(d.schedule_preset)}）`;document.getElementById('s2').textContent=taskLine('抓取任务',d.task_status?.crawler);document.getElementById('s3').textContent=taskLine('情报任务',d.task_status?.intel);renderProgress(d.task_status?.crawler,'crawlerBar','crawlerBarText','抓取任务');renderProgress(d.task_status?.intel,'intelBar','intelBarText','情报任务');const highlights=extractHighlights(d.task_status?.crawler?.message||'');document.getElementById('s4').textContent=`最近报告：${(d.reports||[])[0]||'暂无'}${highlights.length?` | 最近执行重点：${highlights.slice(-3).join(' / ')}`:''}`;}
async function runCrawler(){w('已开始：抓取热点 + RSS，并在结束后自动刷新情报。');const d=await api('/api/run/crawler','POST',{});w(d.ok?'抓取任务已完成。':'抓取任务失败。');if(d.output){extractHighlights(d.output).forEach(x=>w(' - '+x));}await status();}
async function runIntel(){const top=parseInt(document.getElementById('topk').value||'30',10);w(`已开始：情报分析（Top ${top}）。`);const d=await api('/api/run/intel','POST',{top});if(!d.ok){w('情报分析失败：'+ (d.error||''));return;}w(`情报分析完成：命中 ${d.count} 条，结果文件 ${d.json_file||'-'}`);const s=d.summary||{};const src=(s.top_sources||[]).map(x=>`${x.name}(${x.count})`).join(' / ');const kw=(s.top_keywords||[]).map(x=>`${x.name}(${x.count})`).join(' / ');if(src)w(`来源分布：${src}`);if(kw)w(`关键词分布：${kw}`);if(d.ai_summary)w(`AI 总结：${d.ai_summary}`);await status();}
status(); setInterval(status, 2000);"""
    return _html("TrendRadar 任务", "dashboard", body, script)


def _page_intel() -> str:
    body = """<div class='card'><div class='t'>情报分析说明</div>
<div class='hint'>系统会对每条信息按以下维度加权：关键词分组命中、政策/资本来源命中、窗口信号命中、热榜排名加成。<br>你可以直接在本页点“有价值/一般/无价值”反馈，系统会自动学习并调整后续推送优先级。</div></div>
<div class='card'><div class='t'>情报结果（历史流）</div><div class='hint'>新抓取后，旧文章会继续保留。点击标题会标记为已读并变灰。<br>“为什么命中”展示分析结果；“分析细节”可查看打分构成。</div>
<table><thead><tr><th>#</th><th>标题</th><th>来源</th><th>评分</th><th>为什么命中</th><th>分析细节</th><th>反馈</th><th>操作</th></tr></thead><tbody id='rows'></tbody></table></div>"""
    script = """
let lockMap = {};
function rowKey(row){const it=row.item||{};return [String(it.source_name||it.source_id||'').toLowerCase().trim(),String(it.title||'').toLowerCase().trim(),String(it.url||'').toLowerCase().trim()].join('|');}
function goFeedback(row){localStorage.setItem('trendradar_selected_intel',JSON.stringify(row));location.href='/feedback';}
async function qfb(row,rating){const it=row.item||{};const payload={rating,keywords:row.matched_keywords||[],sources:[it.source_name||it.source_id||''].filter(Boolean),add_keywords:[],block_keywords:[],note:'quick_feedback',item_title:it.title||'',item_source:it.source_name||it.source_id||'',item_url:it.url||''};const d=await api('/api/feedback','POST',payload);if(!d.ok){console.warn('反馈失败',d.error||'');return;}await load();}
function reason(row){const a=[];if((row.matched_buckets||[]).length)a.push('关键词组:'+row.matched_buckets.slice(0,2).join('/'));if((row.matched_capital||[]).length)a.push('资本:'+row.matched_capital.slice(0,2).join('/'));if((row.matched_signals||[]).length)a.push('窗口:'+row.matched_signals.slice(0,2).join('/'));if((row.matched_keywords||[]).length)a.push('词:'+row.matched_keywords.slice(0,3).join(','));return a.join(' | ')||'-';}
function detailText(row){const d=row.detail||{};const b=Object.entries(d.bucket_scores||{}).map(x=>x[0]+':'+x[1]).slice(0,3);const s=Object.entries(d.source_scores||{}).map(x=>x[0]+':'+x[1]).slice(0,3);const g=Object.entries(d.signal_scores||{}).map(x=>x[0]+':'+x[1]).slice(0,3);const parts=[];if(b.length)parts.push('关键词分:'+b.join(','));if(s.length)parts.push('来源分:'+s.join(','));if(g.length)parts.push('信号分:'+g.join(','));return parts.join(' | ')||'-';}
async function markRead(key){await api('/api/intel/mark-read','POST',{key});refreshIntelBadge();}
async function load(){const lockData=await api('/api/feedback/locks');lockMap=lockData.items||{};const d=await api('/api/intel/feed');const rows=(d.items||[]).slice().sort((a,b)=>String(b.last_seen||'').localeCompare(String(a.last_seen||'')));const tb=document.getElementById('rows');if(!rows.length){tb.innerHTML='<tr><td colspan=\"8\">暂无数据，请先在“运行任务”页执行情报分析</td></tr>';return;}tb.innerHTML='';rows.forEach((wrap,i)=>{const r=wrap.row||{};const it=r.item||{};const k=wrap.key||rowKey(r);const locked=!!lockMap[k];const read=!!wrap.read;const titleStyle=read?\"style='color:#94a3b8'\":\"\";const fbCell=locked?`<span class='pill'>已反馈</span>`:`<button class='good soft' onclick='qfb(${JSON.stringify(r).replace(/'/g,\"\\\\'\")},5)'>有价值</button> <button class='mid soft' onclick='qfb(${JSON.stringify(r).replace(/'/g,\"\\\\'\")},3)'>一般</button> <button class='bad soft' onclick='qfb(${JSON.stringify(r).replace(/'/g,\"\\\\'\")},1)'>无价值</button>`;const detailBtn=locked?`<button class='soft' disabled style='opacity:.5;cursor:not-allowed'>已锁定</button>`:`<button class='soft' onclick='goFeedback(${JSON.stringify(r).replace(/'/g,\"\\\\'\")})'>详细反馈</button>`;const tr=document.createElement('tr');tr.innerHTML=`<td>${i+1}</td><td><a class='lk' ${titleStyle} href='${esc(it.url||'#')}' target='_blank' onclick='markRead(\"${esc(k)}\")'>${esc(it.title||'')}</a><div class='meta'>首次:${wrap.first_seen||'-'} | 最近:${wrap.last_seen||'-'} | ${read?'已读':'未读'}</div></td><td>${esc(it.source_name||it.source_id||'')}</td><td>${Number(r.score||0).toFixed(2)}</td><td>${esc(reason(r))}</td><td>${esc(detailText(r))}</td><td>${fbCell}</td><td>${detailBtn}</td>`;tb.appendChild(tr);});}
load();"""
    return _html("TrendRadar 情报", "intel", body, script)


def _page_feedback() -> str:
    body = """
<div class='card'><div class='t'>用户反馈</div><div class='hint'>第 8 步：按下面三步操作即可。<br>1) 选择一条情报；2) 判断是否有价值；3) 点一个按钮提交。<br>按钮含义：<b>有价值(5)</b>、<b>一般(3)</b>、<b>无价值(1)</b>。</div>
<div class='row'><div class='col'><div class='meta'>条目选择</div><select id='pick'></select></div></div>
<div id='picked' class='meta'>未选择</div>
<div class='row'><button id='b5' class='good soft' onclick='submit(5)'>有价值(5)</button><button id='b3' class='mid soft' onclick='submit(3)'>一般(3)</button><button id='b1' class='bad soft' onclick='submit(1)'>无价值(1)</button></div>
<div class='row'><div class='col'><div class='meta'>新增关键词(逗号分隔)</div><input id='addk' placeholder='雷击火预警,碳通量监测网络'></div><div class='col'><div class='meta'>屏蔽关键词(逗号分隔)</div><input id='blk' placeholder='营销活动,娱乐八卦'></div></div>
<div class='row'><div class='col'><div class='meta'>备注</div><textarea id='note'></textarea></div></div>
<div class='t'>历史反馈</div>
<div class='hint'>显示最近反馈记录。支持“撤回锁定”或“删除反馈”；撤回/删除都不会回滚已学习权重。</div>
<table><thead><tr><th>时间</th><th>条目</th><th>反馈结论</th><th>备注</th><th>操作</th></tr></thead><tbody id='hist'></tbody></table>
<div id='log' class='log'>请选择一条情报并提交反馈，结果会在这里显示。</div></div>"""
    script = """
let rows=[]; let lockMap={}; function csv(v){return String(v||'').split(',').map(x=>x.trim()).filter(Boolean);}
function w(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent=`[${t}] ${m}\\n`+e.textContent;}
function cur(){const idx=parseInt(document.getElementById('pick').value||'-1',10);return idx>=0?rows[idx]:null;}
function keyOf(r){const it=(r||{}).item||{};return [String(it.source_name||it.source_id||'').toLowerCase().trim(),String(it.title||'').toLowerCase().trim(),String(it.url||'').toLowerCase().trim()].join('|');}
function rateLabel(v){const n=parseInt(v||0,10);if(n>=5)return '有价值';if(n>=3)return '一般';return '无价值';}
function setBtns(disabled){['b5','b3','b1'].forEach(id=>{const el=document.getElementById(id);el.disabled=disabled;});}
function show(){const r=cur();const it=(r||{}).item||{};const locked=!!lockMap[keyOf(r)];document.getElementById('picked').textContent=it.title?`${it.title} | 来源:${it.source_name||it.source_id||'-'} ${locked?'| 状态: 已反馈（不可再次提交）':'| 状态: 未反馈'}`:'未选择';setBtns(locked||!it.title);}
async function load(){const ld=await api('/api/feedback/locks');lockMap=ld.items||{};const d=await api('/api/intel/latest');rows=d.items||[];const s=document.getElementById('pick');s.innerHTML='';rows.forEach((r,i)=>{const it=r.item||{};const locked=!!lockMap[keyOf(r)];const o=document.createElement('option');o.value=String(i);o.textContent=`${i+1}. ${(it.title||'').slice(0,80)}${locked?' [已反馈]':''}`;s.appendChild(o);});if(!rows.length)s.innerHTML=\"<option value='-1'>暂无情报数据</option>\";const p=localStorage.getItem('trendradar_selected_intel');if(p){try{const row=JSON.parse(p);const idx=rows.findIndex(x=>(x.item||{}).title===(row.item||{}).title);if(idx>=0)s.value=String(idx);}catch(_){}}show();await loadHistory();}
async function submit(rating){const r=cur();if(!r){w('请先选择条目');return;}if(lockMap[keyOf(r)]){w('该条已反馈，不能重复提交');return;}const it=r.item||{};const payload={rating,keywords:r.matched_keywords||[],sources:[it.source_name||it.source_id||''].filter(Boolean),add_keywords:csv(document.getElementById('addk').value),block_keywords:csv(document.getElementById('blk').value),note:document.getElementById('note').value||'',item_title:it.title||'',item_source:it.source_name||it.source_id||'',item_url:it.url||''};const d=await api('/api/feedback','POST',payload);w(d.ok?`反馈成功(${rating}分)`:`反馈失败:${d.error||''}`);if(d.ok)await load();}
async function undoLock(title,source,url){const d=await api('/api/feedback/undo','POST',{item_title:title,item_source:source,item_url:url});w(d.ok?'已撤销锁定，可重新反馈':'撤销失败:'+ (d.error||''));if(d.ok)await load();}
async function deleteFeedback(eventIndex,title,source,url){if(!confirm('确认删除这条反馈记录？'))return;const d=await api('/api/feedback/delete','POST',{event_index:eventIndex,item_title:title,item_source:source,item_url:url});w(d.ok?'已删除反馈记录':'删除失败:'+ (d.error||''));if(d.ok)await load();}
async function loadHistory(){const d=await api('/api/feedback/history');const rows=d.events||[];const tb=document.getElementById('hist');tb.innerHTML='';if(!rows.length){tb.innerHTML='<tr><td colspan=\"5\">暂无历史反馈</td></tr>';return;}rows.forEach(ev=>{const title=String(ev.item_title||'').trim();const source=String(ev.item_source||'').trim();const url=ev.item_url||'';const note=String(ev.note||'').trim();const idx=parseInt(ev.event_index??-1,10);const itemText=title?`${title}${source?`（${source}）`:''}`:'未绑定条目';const actions=title?`<button class='soft' onclick='undoLock(${JSON.stringify(title)},${JSON.stringify(source)},${JSON.stringify(url)})'>撤回锁定</button> <button class='bad soft' onclick='deleteFeedback(${idx},${JSON.stringify(title)},${JSON.stringify(source)},${JSON.stringify(url)})'>删除反馈</button>`:(idx>=0?`<button class='bad soft' onclick='deleteFeedback(${idx},"","","")'>删除反馈</button>`:'-');const tr=document.createElement('tr');tr.innerHTML=`<td>${ev.time||'-'}</td><td>${esc(itemText)}</td><td>${rateLabel(ev.rating||0)}</td><td>${esc(note||'-')}</td><td>${actions}</td>`;tb.appendChild(tr);});}
document.getElementById('pick').addEventListener('change',show); load();"""
    return _html("TrendRadar 用户反馈", "feedback", body, script)


def _page_keywords() -> str:
    body = """
<div class='card'><div class='t'>关键词管理（按业务流程）</div>
<div class='hint'>建议按流程拆分：线索发现、政策研判、资本动向、合作机会。每个分组可配置权重与关键词列表。</div>
<div class='meta'><span class='pill'>高优先级配置</span> 修改后会直接影响情报命中和排序，请及时保存。</div>
<div class='row'><button onclick='addBucket()'>新增分组</button><button class='alt' onclick='save()'>保存关键词配置</button><span id='dirty' class='pill' style='display:none;background:#fff7ed;border-color:#fdba74;color:#9a3412'>有未保存修改</span></div>
<div class='t'>分组总览</div>
<div class='hint'>先在总览里看分组结构，再在下方逐个编辑内容。</div>
<table><thead><tr><th>#</th><th>分组名</th><th>流程阶段</th><th>权重</th><th>关键词数</th><th>操作</th></tr></thead><tbody id='bucketSummary'></tbody></table>
<div class='hint'>详细编辑区</div>
<div id='boxes'></div>
<div class='t'>候选关键词（来自已接受反馈）</div>
<div class='hint'>系统会自动从“有价值/一般偏高”反馈中提炼候选词（规则+可选 LLM），并仅展示高相关候选。勾选后可一键加入指定分组。</div>
<div class='row'><div class='col'><div class='meta'>加入到分组</div><select id='suggestBucket'></select></div><div class='col'><div class='meta'>操作</div><button class='alt' onclick='refreshSuggestions()'>刷新候选</button> <button onclick='applySuggestions()'>加入选中关键词</button></div></div>
<table><thead><tr><th>选择</th><th>关键词</th><th>来源</th><th>说明</th></tr></thead><tbody id='suggestRows'></tbody></table>
<div id='log' class='log'>关键词配置日志会显示在这里。</div></div>"""
    script = """
let buckets=[]; let dirty=false; let suggestions=[];
function w(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent=`[${t}] ${m}\\n`+e.textContent;}
function setDirty(v=true){dirty=v;document.getElementById('dirty').style.display=dirty?'inline-block':'none';}
function stageSelect(stage,idx){const opts=['线索发现','政策研判','资本动向','合作机会','观察'];return `<select data-k='flow_stage' data-i='${idx}'>${opts.map(x=>`<option value='${x}' ${x===stage?'selected':''}>${x}</option>`).join('')}</select>`;}
function summary(){const tb=document.getElementById('bucketSummary');tb.innerHTML='';if(!buckets.length){tb.innerHTML='<tr><td colspan=\"6\">暂无分组，请点击“新增分组”。</td></tr>';return;}buckets.forEach((b,idx)=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${idx+1}</td><td>${esc(b.name||'-')}</td><td>${esc(b.flow_stage||'观察')}</td><td>${Number(b.weight||1).toFixed(2)}</td><td>${(b.keywords||[]).length}</td><td><button class='soft' onclick='jumpTo(${idx})'>定位编辑</button></td>`;tb.appendChild(tr);});}
function renderSuggestBuckets(){const s=document.getElementById('suggestBucket');s.innerHTML='';if(!buckets.length){s.innerHTML=\"<option value='-1'>暂无分组</option>\";return;}buckets.forEach((b,idx)=>{const o=document.createElement('option');o.value=String(idx);o.textContent=`${idx+1}. ${b.name||('分组'+(idx+1))}`;s.appendChild(o);});}
function renderSuggestions(){const tb=document.getElementById('suggestRows');tb.innerHTML='';if(!suggestions.length){tb.innerHTML='<tr><td colspan=\"4\">暂无候选关键词（请先产生“有价值”反馈）</td></tr>';return;}suggestions.forEach((x,i)=>{const tr=document.createElement('tr');tr.innerHTML=`<td><input type='checkbox' id='sg-${i}' checked></td><td>${esc(x.keyword||'')}</td><td>${esc(x.source||'-')}</td><td>${esc(x.reason||'-')}</td>`;tb.appendChild(tr);});}
function render(){const root=document.getElementById('boxes');root.innerHTML='';if(!buckets.length){root.innerHTML='<div class=\"meta\">暂无分组，点“新增分组”。</div>';summary();renderSuggestBuckets();renderSuggestions();return;}buckets.forEach((b,idx)=>{const box=document.createElement('div');box.className='card';box.id=`bucket-${idx}`;box.innerHTML=`<div class='row'><div class='col'><div class='meta'>分组名</div><input data-k='name' data-i='${idx}' value='${esc(b.name||'')}'></div><div class='col'><div class='meta'>业务流程阶段</div>${stageSelect(b.flow_stage||'观察',idx)}</div><div class='col'><div class='meta'>权重</div><input data-k='weight' data-i='${idx}' value='${Number(b.weight||1).toFixed(2)}'></div></div><div class='meta'>关键词（每行一个）</div><textarea data-k='keywords' data-i='${idx}'>${esc((b.keywords||[]).join('\\n'))}</textarea><div class='row'><span class='pill'>关键词数: ${(b.keywords||[]).length}</span><button class='bad soft' onclick='removeBucket(${idx})'>删除分组</button></div>`;root.appendChild(box);});Array.from(root.querySelectorAll('input,textarea,select')).forEach(el=>{el.addEventListener('change',onChange);});summary();renderSuggestBuckets();renderSuggestions();}
function jumpTo(idx){const el=document.getElementById(`bucket-${idx}`);if(el){el.scrollIntoView({behavior:'smooth',block:'start'});}}
function onChange(e){const i=parseInt(e.target.getAttribute('data-i')||'-1',10);const k=e.target.getAttribute('data-k');if(i<0||!k)return;let v=e.target.value;if(k==='weight')v=parseFloat(v||'1')||1;if(k==='keywords')v=String(v||'').split('\\n').map(x=>x.trim()).filter(Boolean);buckets[i][k]=v;setDirty(true);}
function addBucket(){buckets.push({name:'新分组',flow_stage:'线索发现',weight:1.0,keywords:[]});setDirty(true);render();}
function removeBucket(i){if(!confirm('确认删除该分组？'))return;buckets.splice(i,1);setDirty(true);render();}
async function refreshSuggestions(){const d=await api('/api/keywords/suggestions');if(!d.ok){w('候选词提炼失败:'+ (d.error||''));return;}suggestions=d.items||[];renderSuggestions();w(`候选词刷新完成：${suggestions.length} 条（已接受反馈 ${d.accepted_count||0} 条，LLM ${d.used_llm?'已参与':'未参与'}）`);}
async function load(){const d=await api('/api/settings/keywords');buckets=d.keyword_buckets||[];render();await refreshSuggestions();}
function applySuggestions(){if(!buckets.length){w('请先创建一个关键词分组');return;}const target=parseInt(document.getElementById('suggestBucket').value||'-1',10);if(target<0||target>=buckets.length){w('请选择有效分组');return;}const selected=[];suggestions.forEach((x,i)=>{const el=document.getElementById(`sg-${i}`);if(el&&el.checked)selected.push(String(x.keyword||'').trim());});if(!selected.length){w('请先勾选候选关键词');return;}const set=new Set((buckets[target].keywords||[]).map(x=>String(x).trim()).filter(Boolean));let added=0;selected.forEach(k=>{if(k&&!set.has(k)){set.add(k);added++;}});buckets[target].keywords=Array.from(set);setDirty(true);render();w(`已加入 ${added} 个关键词到分组“${buckets[target].name||('分组'+(target+1))}”，请点击“保存关键词配置”生效`);}
async function save(){const d=await api('/api/settings/keywords','POST',{keyword_buckets:buckets});if(d.ok){w('保存成功');setDirty(false);}else w('保存失败:'+ (d.error||''));}
load();"""
    return _html("TrendRadar 关键词", "keywords", body, script)


def _page_llm() -> str:
    body = """
<div class='card'><div class='t'>大模型配置</div>
<div class='hint'>用于主流程 AI 分析能力（运行抓取时生效）。建议先配置模型与 API，再在配置中开启分析。</div>
<div class='row'><div class='col'><div class='meta'>模型名 (ai.model)</div><input id='model' placeholder='openai/gpt-4o-mini'></div><div class='col'><div class='meta'>API Base (ai.api_base)</div><input id='api_base' placeholder='https://api.openai.com/v1'></div></div>
<div class='row'><div class='col'><div class='meta'>API Key (ai.api_key)</div><input id='api_key' placeholder='sk-...'></div><div class='col'><div class='meta'>Timeout 秒</div><input id='timeout' value='120'></div></div>
<div class='row'><div class='col'><div class='meta'>Temperature</div><input id='temperature' value='1.0'></div><div class='col'><div class='meta'>Max Tokens</div><input id='max_tokens' value='5000'></div></div>
<div class='row'><div class='col'><div class='meta'>启用 AI 分析</div><select id='analysis_enabled'><option value='true'>开启</option><option value='false'>关闭</option></select></div><div class='col'><div class='meta'>AI 分析范围</div><select id='analysis_mode'><option value='follow_report'>跟随报告模式</option><option value='daily'>固定为当日汇总</option><option value='current'>固定为当前榜单</option><option value='incremental'>固定为仅新增</option></select></div><div class='col'><div class='meta'>分析输出语言</div><input id='analysis_language' value='Chinese'></div></div>
<div class='meta'>说明：AI 分析范围决定模型读取哪一批新闻；“跟随报告模式”表示与当前推送模式保持一致。</div>
<div class='row'><button onclick='save()'>保存模型配置</button></div>
<div class='hint'>安全提示：API Key 仅保存在本地 `output/webapp/settings.json`。</div>
<div id='log' class='log'>模型配置日志会显示在这里。</div></div>"""
    script = """
function w(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent=`[${t}] ${m}\\n`+e.textContent;}
function normalizeMode(v){if(v==='always_analyze'||v==='always_skip')return 'follow_report';return v||'follow_report';}
async function load(){const d=await api('/api/settings/llm');const v=d.llm||{};document.getElementById('model').value=v.model||'';document.getElementById('api_base').value=v.api_base||'';document.getElementById('api_key').value=v.api_key||'';document.getElementById('timeout').value=v.timeout||120;document.getElementById('temperature').value=v.temperature||1.0;document.getElementById('max_tokens').value=v.max_tokens||5000;document.getElementById('analysis_enabled').value=String(!!v.analysis_enabled);document.getElementById('analysis_mode').value=normalizeMode(v.analysis_mode);document.getElementById('analysis_language').value=v.analysis_language||'Chinese';}
async function save(){const payload={llm:{model:document.getElementById('model').value.trim(),api_base:document.getElementById('api_base').value.trim(),api_key:document.getElementById('api_key').value.trim(),timeout:parseInt(document.getElementById('timeout').value||'120',10),temperature:parseFloat(document.getElementById('temperature').value||'1.0'),max_tokens:parseInt(document.getElementById('max_tokens').value||'5000',10),analysis_enabled:document.getElementById('analysis_enabled').value==='true',analysis_mode:document.getElementById('analysis_mode').value,analysis_language:document.getElementById('analysis_language').value.trim()}};const d=await api('/api/settings/llm','POST',payload);w(d.ok?'保存成功':'保存失败:'+ (d.error||''));}
load();"""
    return _html("TrendRadar 大模型", "llm", body, script)


def _page_sources() -> str:
    body = """
<div class='card'><div class='t'>数据源管理</div>
<div class='hint'>支持管理热榜平台与 RSS 源。可“自动发现”历史抓取中出现的源，或手工新增。</div>
<div class='row'><button onclick='discover()'>自动发现数据源</button><button class='alt' onclick='bootstrap()'>一键补全主流源</button><button class='alt' onclick='save()'>保存数据源配置</button></div>
<div id='discoverInfo' class='meta'>尚未执行自动发现</div>
<div class='row'><div class='col'><div class='meta'>热榜总开关</div><select id='platformsEnabled'><option value='true'>开启</option><option value='false'>关闭</option></select></div><div class='col'><div class='meta'>RSS 总开关</div><select id='rssEnabled'><option value='true'>开启</option><option value='false'>关闭</option></select></div></div>
<div class='row'><div class='col'><div class='t'>热榜平台</div><div id='platforms'></div><button onclick='addPlatform()'>新增平台</button></div><div class='col'><div class='t'>RSS 源</div><div id='feeds'></div><button onclick='addFeed()'>新增 RSS</button></div></div>
<div id='log' class='log'>数据源配置日志会显示在这里。</div></div>
<div class='card'><div class='t'>热榜平台添加与维护说明</div>
<div class='hint'><b>热榜平台</b>指网站/应用的实时热门榜单抓取源（例如：微博热搜、知乎热榜、抖音热点）。</div>
<div class='hint'>推荐维护流程：<br>1) 点击“<b>一键补全主流源</b>”初始化常见平台；<br>2) 点击“<b>自动发现数据源</b>”导入你历史数据中真实出现的平台；<br>3) 需要新增时，在“热榜平台”区域填写平台 <b>id</b>（newsnow 支持的源）和显示名称；<br>4) 点“保存数据源配置”后，到“运行任务”执行一次抓取验证。</div>
<div class='hint'>常见平台 ID 示例：<span id='platformExamples'>加载中...</span></div>
<div class='hint'>提示：如不确定 id，可先补全主流源再按需删减，避免手输出错。</div></div>"""
    script = """
let state={platforms_enabled:true,platforms:[],rss_enabled:true,rss_feeds:[]};
function w(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent=`[${t}] ${m}\\n`+e.textContent;}
async function loadExamples(){try{const d=await api('/api/sources/recommended','GET',null,false);const arr=(d.recommended?.platforms||[]).slice(0,12).map(x=>x.id);document.getElementById('platformExamples').textContent=arr.length?arr.join('、'):'暂无';}catch(_){document.getElementById('platformExamples').textContent='暂无';}}
function render(){document.getElementById('platformsEnabled').value=String(!!state.platforms_enabled);document.getElementById('rssEnabled').value=String(!!state.rss_enabled);const p=document.getElementById('platforms');const f=document.getElementById('feeds');p.innerHTML='';f.innerHTML='';(state.platforms||[]).forEach((x,i)=>{const d=document.createElement('div');d.className='row';d.innerHTML=`<input data-t='p' data-i='${i}' data-k='id' value='${esc(x.id||'')}' placeholder='id' style='max-width:180px'><input data-t='p' data-i='${i}' data-k='name' value='${esc(x.name||'')}' placeholder='名称'><button class='bad' onclick='delP(${i})'>删</button>`;p.appendChild(d);});(state.rss_feeds||[]).forEach((x,i)=>{const d=document.createElement('div');d.className='row';d.innerHTML=`<input data-t='f' data-i='${i}' data-k='id' value='${esc(x.id||'')}' placeholder='id' style='max-width:140px'><input data-t='f' data-i='${i}' data-k='name' value='${esc(x.name||'')}' placeholder='名称'><input data-t='f' data-i='${i}' data-k='url' value='${esc(x.url||'')}' placeholder='RSS URL'><button class='bad' onclick='delF(${i})'>删</button>`;f.appendChild(d);});Array.from(document.querySelectorAll('#platforms input,#feeds input')).forEach(el=>el.onchange=onChange);}
function onChange(e){const i=parseInt(e.target.dataset.i||'-1',10),k=e.target.dataset.k,t=e.target.dataset.t;if(i<0)return;const v=e.target.value; if(t==='p')state.platforms[i][k]=v; else state.rss_feeds[i][k]=v;}
function addPlatform(){state.platforms.push({id:'',name:''});render();}
function addFeed(){state.rss_feeds.push({id:'',name:'',url:''});render();}
function delP(i){state.platforms.splice(i,1);render();}
function delF(i){state.rss_feeds.splice(i,1);render();}
async function load(){const d=await api('/api/settings/sources');state=d.sources||state;render();}
async function bootstrap(){const d=await api('/api/sources/bootstrap','POST',{});if(!d.ok){w('补全失败:'+ (d.error||''));return;}state=d.sources||state;render();w(`已补全：新增平台 ${d.added_platforms||0} 个，新增 RSS ${d.added_rss||0} 个`);}
async function discover(){const d=await api('/api/sources/discover');if(!d.ok){w('发现失败');return;}const dp=d.discovered.platforms||[];const df=d.discovered.rss_feeds||[];let addedP=0,addedF=0;const pset=new Set((state.platforms||[]).map(x=>x.id));dp.forEach(x=>{if(x.id&&!pset.has(x.id)){state.platforms.push({id:x.id,name:x.name||x.id});pset.add(x.id);addedP++;}});const fset=new Set((state.rss_feeds||[]).map(x=>x.id));df.forEach(x=>{if(x.id&&!fset.has(x.id)){state.rss_feeds.push({id:x.id,name:x.name||x.id,url:x.url||''});fset.add(x.id);addedF++;}});render();document.getElementById('discoverInfo').textContent=`发现平台 ${dp.length} 个（新增 ${addedP}），发现 RSS ${df.length} 个（新增 ${addedF}）`;if(addedP===0&&addedF===0)w('自动发现完成：没有新增，可能都已在配置中');else w(`自动发现完成：新增平台 ${addedP}，新增 RSS ${addedF}`);}
async function save(){state.platforms_enabled=document.getElementById('platformsEnabled').value==='true';state.rss_enabled=document.getElementById('rssEnabled').value==='true';state.platforms=(state.platforms||[]).map(x=>({id:(x.id||'').trim(),name:(x.name||'').trim()})).filter(x=>x.id);state.rss_feeds=(state.rss_feeds||[]).map(x=>({id:(x.id||'').trim(),name:(x.name||'').trim(),url:(x.url||'').trim()})).filter(x=>x.id&&x.url);const d=await api('/api/settings/sources','POST',{sources:state});w(d.ok?'保存成功':'保存失败:'+ (d.error||''));if(d.ok)state=d.sources;}
load();loadExamples();"""
    return _html("TrendRadar 数据源", "sources", body, script)


def _page_notification() -> str:
    body = """
<div class='card'><div class='t'>通知配置</div><div class='hint'>保存到 output/webapp/settings.json，抓取时自动注入运行时配置。<br>支持：企业微信、飞书、钉钉、Telegram、邮件、Slack、通用 Webhook。</div>
<div class='row'><div class='col'><div class='meta'>通知总开关</div><select id='enabled'><option value='true'>开启</option><option value='false'>关闭</option></select></div><div class='col'><div class='meta'>企业微信消息格式</div><select id='wtype'><option value='markdown'>Markdown（群机器人）</option><option value='text'>纯文本（个人应用）</option></select></div></div>
<div class='row'><div class='col'><div class='meta'>企业微信 webhook</div><input id='wework'></div><div class='col'><div class='meta'>飞书 webhook</div><input id='feishu'></div></div>
<div class='row'><div class='col'><div class='meta'>钉钉 webhook</div><input id='dingtalk'></div><div class='col'><div class='meta'>Slack webhook</div><input id='slack'></div></div>
<div class='row'><div class='col'><div class='meta'>Telegram token</div><input id='ttoken'></div><div class='col'><div class='meta'>Telegram chat_id</div><input id='tchat'></div></div>
<div class='row'><div class='col'><div class='meta'>邮件发件人(email.from)</div><input id='emailFrom' placeholder='you@example.com'></div><div class='col'><div class='meta'>邮件授权码(email.password)</div><input id='emailPassword' placeholder='授权码/SMTP密码'></div></div>
<div class='row'><div class='col'><div class='meta'>邮件收件人(email.to)</div><input id='emailTo' placeholder='a@xx.com,b@xx.com'></div><div class='col'><div class='meta'>SMTP 服务器(email.smtp_server)</div><input id='emailSmtpServer' placeholder='smtp.qq.com'></div></div>
<div class='row'><div class='col'><div class='meta'>SMTP 端口(email.smtp_port)</div><input id='emailSmtpPort' placeholder='465'></div></div>
<div class='row'><div class='col'><div class='meta'>通用 webhook</div><input id='generic'></div></div>
<div class='hint'>安全提示：请勿把 webhook/token/邮箱授权码提交到 git 仓库。</div>
<div class='row'><button onclick='save()'>保存配置</button></div><div id='log' class='log'>通知配置日志会显示在这里。</div></div>"""
    script = """
function w(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent=`[${t}] ${m}\\n`+e.textContent;}
async function load(){const d=await api('/api/settings/notification');const n=d.notification||{};document.getElementById('enabled').value=String(!!n.enabled);document.getElementById('wtype').value=n.wework_msg_type||'markdown';document.getElementById('wework').value=n.wework_webhook_url||'';document.getElementById('feishu').value=n.feishu_webhook_url||'';document.getElementById('dingtalk').value=n.dingtalk_webhook_url||'';document.getElementById('slack').value=n.slack_webhook_url||'';document.getElementById('ttoken').value=n.telegram_bot_token||'';document.getElementById('tchat').value=n.telegram_chat_id||'';document.getElementById('emailFrom').value=n.email_from||'';document.getElementById('emailPassword').value=n.email_password||'';document.getElementById('emailTo').value=n.email_to||'';document.getElementById('emailSmtpServer').value=n.email_smtp_server||'';document.getElementById('emailSmtpPort').value=n.email_smtp_port||'';document.getElementById('generic').value=n.generic_webhook_url||'';}
async function save(){const payload={notification:{enabled:document.getElementById('enabled').value==='true',wework_msg_type:document.getElementById('wtype').value,wework_webhook_url:document.getElementById('wework').value.trim(),feishu_webhook_url:document.getElementById('feishu').value.trim(),dingtalk_webhook_url:document.getElementById('dingtalk').value.trim(),slack_webhook_url:document.getElementById('slack').value.trim(),telegram_bot_token:document.getElementById('ttoken').value.trim(),telegram_chat_id:document.getElementById('tchat').value.trim(),email_from:document.getElementById('emailFrom').value.trim(),email_password:document.getElementById('emailPassword').value.trim(),email_to:document.getElementById('emailTo').value.trim(),email_smtp_server:document.getElementById('emailSmtpServer').value.trim(),email_smtp_port:document.getElementById('emailSmtpPort').value.trim(),generic_webhook_url:document.getElementById('generic').value.trim()}};const d=await api('/api/settings/notification','POST',payload);w(d.ok?'保存成功':'保存失败:'+ (d.error||''));}
load();"""
    return _html("TrendRadar 通知", "notification", body, script)


def _page_schedule() -> str:
    body = """
<div class='card'><div class='t'>定时调度配置</div><div class='hint'>设置 schedule.enabled 与 schedule.preset，使用 timeline.yaml 中预设。<br>保存后不会立即触发任务，下一次“执行抓取”会按新调度生效。</div>
<div class='row'><div class='col'><div class='meta'>调度总开关</div><select id='enabled'><option value='true'>开启</option><option value='false'>关闭</option></select></div><div class='col'><div class='meta'>调度类型</div><select id='preset'></select></div></div>
<div id='presetHelp' class='meta'></div>
<div class='row'><button onclick='save()'>保存调度</button></div><div id='log' class='log'>调度配置日志会显示在这里。</div></div>"""
    script = """
function w(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent=`[${t}] ${m}\\n`+e.textContent;}
function explain(p){const m={always_on:'全天候监控，有新增即可推送',morning_evening:'白天监控+早晚节奏推送（推荐）',office_hours:'工作时段重点推送，周末降频',night_owl:'偏夜间节奏，适合晚间复盘',custom:'完全按 timeline.yaml 自定义'};return m[p]||'使用 timeline.yaml 中该 preset 定义';}
function label(p){const m={always_on:'全天候',morning_evening:'早晚节奏（推荐）',office_hours:'工作时段',night_owl:'夜间复盘',custom:'自定义'};return m[p]||p;}
function updateHelp(){const e=document.getElementById('presetHelp');const en=document.getElementById('enabled').value==='true';const p=document.getElementById('preset').value;e.textContent=`当前解释：${en?'已启用调度':'已关闭调度'}；模板 ${p}：${explain(p)}`;}
async function load(){const d=await api('/api/settings/schedule');const s=d.schedule||{};const presets=d.presets||[];document.getElementById('enabled').value=String(!!s.enabled);const p=document.getElementById('preset');p.innerHTML='';presets.forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=`${label(x)}（${x}）`;p.appendChild(o);});if(s.preset&&presets.includes(s.preset))p.value=s.preset;updateHelp();}
async function save(){const payload={schedule:{enabled:document.getElementById('enabled').value==='true',preset:document.getElementById('preset').value}};const d=await api('/api/settings/schedule','POST',payload);w(d.ok?'保存成功，下次抓取按此调度':'保存失败:'+ (d.error||''));}
document.getElementById('enabled').addEventListener('change',updateHelp);
document.getElementById('preset').addEventListener('change',updateHelp);
load();"""
    return _html("TrendRadar 调度", "schedule", body, script)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "TrendRadarWeb/0.4"

    def do_GET(self) -> None:
        p = urlparse(self.path).path
        if p in ("/", "/dashboard"):
            raw = _page_dashboard().encode("utf-8")
        elif p == "/intel":
            raw = _page_intel().encode("utf-8")
        elif p == "/feedback":
            raw = _page_feedback().encode("utf-8")
        elif p == "/keywords":
            raw = _page_keywords().encode("utf-8")
        elif p == "/llm":
            raw = _page_llm().encode("utf-8")
        elif p == "/sources":
            raw = _page_sources().encode("utf-8")
        elif p == "/notification":
            raw = _page_notification().encode("utf-8")
        elif p == "/schedule":
            raw = _page_schedule().encode("utf-8")
        elif p == "/api/status":
            settings = _load_settings()
            intel_rows = _j(_p_intel_json()) or []
            fb = _j(_p_feedback()) or {}
            task = _load_task_status()
            unread = _intel_unread_count()
            _json_response(
                self,
                {
                    "ok": True,
                    "intel_count": len(intel_rows),
                    "intel_unread": unread,
                    "feedback_events": len(fb.get("events", [])),
                    "updated_at": fb.get("updated_at", ""),
                    "reports": _recent_reports(),
                    "notification_configured": _has_notify(settings.get("notification", {})),
                    "schedule_enabled": bool(settings.get("schedule", {}).get("enabled", True)),
                    "schedule_preset": str(settings.get("schedule", {}).get("preset", "morning_evening")),
                    "task_status": task,
                },
            )
            return
        elif p == "/api/intel/latest":
            _json_response(self, {"ok": True, "items": _j(_p_intel_json()) or []})
            return
        elif p == "/api/intel/feed":
            _json_response(self, {"ok": True, **_intel_feed_with_read()})
            return
        elif p == "/api/settings/notification":
            _json_response(self, {"ok": True, "notification": _load_settings().get("notification", {})})
            return
        elif p == "/api/settings/schedule":
            _json_response(self, {"ok": True, "schedule": _load_settings().get("schedule", {}), "presets": _timeline_presets()})
            return
        elif p == "/api/settings/keywords":
            _json_response(self, {"ok": True, "keyword_buckets": _read_keyword_buckets()})
            return
        elif p == "/api/settings/llm":
            _json_response(self, {"ok": True, "llm": _load_settings().get("llm", {})})
            return
        elif p == "/api/settings/sources":
            _json_response(self, {"ok": True, "sources": _load_settings().get("sources", {})})
            return
        elif p == "/api/sources/discover":
            _json_response(self, {"ok": True, "discovered": _discover_sources()})
            return
        elif p == "/api/sources/recommended":
            _json_response(self, {"ok": True, "recommended": _recommended_sources()})
            return
        elif p == "/api/feedback/locks":
            locks = _load_feedback_locks()
            _json_response(self, {"ok": True, "items": _feedback_locks_raw_map(locks), "total": len((locks.get("items") or {}).keys())})
            return
        elif p == "/api/feedback/history":
            state = load_feedback_state(str(_p_feedback()))
            raw_events = state.get("events", [])
            start_idx = max(0, len(raw_events) - 200)
            events = []
            for idx in range(len(raw_events) - 1, start_idx - 1, -1):
                events.append({**(raw_events[idx] or {}), "event_index": idx})
            _json_response(self, {"ok": True, "events": events})
            return
        elif p == "/api/keywords/suggestions":
            _json_response(self, _suggest_keywords_from_feedback(limit=20))
            return
        else:
            _json_response(self, {"ok": False, "error": "Not Found"}, status=404)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        p = urlparse(self.path).path
        try:
            body = _read_json(self)
        except Exception as e:
            _json_response(self, {"ok": False, "error": f"请求体解析失败: {e}"}, status=400)
            return

        if p == "/api/run/crawler":
            ok, output = _run_crawler()
            _json_response(self, {"ok": ok, "output": output[-3000:]})
            return

        if p == "/api/run/intel":
            try:
                top = int(body.get("top", 30))
                _json_response(self, {"ok": True, **_run_intel(max(1, top))})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=500)
            return

        if p == "/api/sources/bootstrap":
            try:
                _json_response(self, _bootstrap_sources_into_settings())
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        if p == "/api/feedback":
            try:
                item_title = str(body.get("item_title", ""))
                item_source = str(body.get("item_source", ""))
                item_url = str(body.get("item_url", ""))
                if item_title:
                    locks = _load_feedback_locks()
                    raw_key = _feedback_raw_key(item_title, item_source, item_url)
                    existing_raw = _feedback_locks_raw_map(locks)
                    if raw_key in existing_raw:
                        _json_response(self, {"ok": False, "error": "该条新闻已反馈，禁止重复提交"}, status=409)
                        return

                st = record_feedback(
                    state_file=str(_p_feedback()),
                    rating=int(body.get("rating", 5)),
                    keywords=body.get("keywords", []),
                    source_names=body.get("sources", []),
                    add_keywords=body.get("add_keywords", []),
                    block_keywords=body.get("block_keywords", []),
                    note=str(body.get("note", "")),
                )
                state = load_feedback_state(str(_p_feedback()))
                if state.get("events"):
                    state["events"][-1]["item_title"] = item_title
                    state["events"][-1]["item_source"] = item_source
                    state["events"][-1]["item_url"] = item_url
                    save_feedback_state(str(_p_feedback()), state)
                if item_title:
                    locks = _load_feedback_locks()
                    items = locks.setdefault("items", {})
                    items[_feedback_key(item_title, item_source, item_url)] = {
                        "title": item_title,
                        "source": item_source,
                        "url": item_url,
                        "rating": int(body.get("rating", 5)),
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }
                    _save_feedback_locks(locks)
                _json_response(self, {"ok": True, "updated_at": st.get("updated_at", ""), "events": len(st.get("events", []))})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        if p == "/api/feedback/undo":
            try:
                item_title = str(body.get("item_title", ""))
                item_source = str(body.get("item_source", ""))
                item_url = str(body.get("item_url", ""))
                if not item_title:
                    _json_response(self, {"ok": False, "error": "缺少 item_title"}, status=400)
                    return
                removed = _remove_feedback_lock(item_title, item_source, item_url)
                _json_response(self, {"ok": True, "removed": bool(removed)})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        if p == "/api/feedback/delete":
            try:
                event_index = int(body.get("event_index", -1))
                state = load_feedback_state(str(_p_feedback()))
                events = state.get("events", [])
                if event_index < 0 or event_index >= len(events):
                    _json_response(self, {"ok": False, "error": "无效 event_index"}, status=400)
                    return
                removed_event = events.pop(event_index)
                state["events"] = events
                state["updated_at"] = datetime.now().isoformat(timespec="seconds")
                save_feedback_state(str(_p_feedback()), state)

                item_title = str(body.get("item_title", "") or removed_event.get("item_title", ""))
                item_source = str(body.get("item_source", "") or removed_event.get("item_source", ""))
                item_url = str(body.get("item_url", "") or removed_event.get("item_url", ""))
                lock_removed = _remove_feedback_lock(item_title, item_source, item_url) if item_title else False
                _json_response(self, {"ok": True, "removed": True, "lock_removed": lock_removed})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        if p == "/api/intel/mark-read":
            try:
                key = str(body.get("key", "")).strip()
                if not key:
                    _json_response(self, {"ok": False, "error": "缺少 key"}, status=400)
                    return
                state = _load_read_state()
                state.setdefault("items", {})[key] = datetime.now().isoformat(timespec="seconds")
                _save_read_state(state)
                _json_response(self, {"ok": True})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        if p == "/api/settings/notification":
            try:
                saved = _save_settings({"notification": body.get("notification", {})})
                _json_response(self, {"ok": True, "notification": saved.get("notification", {})})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        if p == "/api/settings/schedule":
            try:
                saved = _save_settings({"schedule": body.get("schedule", {})})
                _json_response(self, {"ok": True, "schedule": saved.get("schedule", {})})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        if p == "/api/settings/keywords":
            try:
                data = _save_keyword_buckets(body.get("keyword_buckets", []))
                _json_response(self, {"ok": True, "keyword_buckets": data})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        if p == "/api/settings/llm":
            try:
                saved = _save_settings({"llm": body.get("llm", {})})
                _json_response(self, {"ok": True, "llm": saved.get("llm", {})})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        if p == "/api/settings/sources":
            try:
                saved = _save_settings({"sources": body.get("sources", {})})
                _json_response(self, {"ok": True, "sources": saved.get("sources", {})})
            except Exception as e:
                _json_response(self, {"ok": False, "error": str(e)}, status=400)
            return

        _json_response(self, {"ok": False, "error": "Not Found"}, status=404)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")


def run_server(host: str = "127.0.0.1", port: int = 8899) -> None:
    s = ThreadingHTTPServer((host, port), AppHandler)
    print(f"TrendRadar Web 控制台已启动: http://{host}:{port}")
    s.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TrendRadar 本地 Web 控制台")
    p.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    p.add_argument("--port", type=int, default=8899, help="监听端口，默认 8899")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run_server(args.host, args.port)
    return 0
