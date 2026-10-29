"""Deterministic report scoring for fixed PaperScope benchmarks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PLACEHOLDER_PATTERNS = (
    r"待模型",
    r"正在依据",
    r"尚未配置",
    r"未收录摘要",
    r"需要进一步",
    r"当前材料(?:未|不足以)",
    r"仍需.*核验",
    r"未能通过校验",
)


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def score_report(report: dict, benchmark: dict) -> dict:
    """Return a 100-point, inspectable score without calling any model."""
    categories: dict[str, dict[str, Any]] = {}

    paper = report.get("paper") if isinstance(report.get("paper"), dict) else {}
    metadata_checks = {
        "题名匹配": _normalized(benchmark.get("title")) in _normalized(paper.get("title")),
        "年份存在": bool(paper.get("year")),
        "作者存在": _useful_text(paper.get("authors"), 4),
        "来源链接存在": str(paper.get("sourceUrl") or "").startswith("http"),
        "标识符存在": _useful_text(paper.get("identifier"), 5),
    }
    categories["metadata"] = _category("论文识别", 10, metadata_checks)

    analysis = report.get("analysis") if isinstance(report.get("analysis"), dict) else {}
    synthesis = report.get("synthesis") if isinstance(report.get("synthesis"), dict) else {}
    structure_checks = {
        "一句话定位": _useful_text(paper.get("oneSentence"), 20),
        "研究问题": _useful_text(analysis.get("problem"), 60),
        "至少三项贡献": _useful_list(analysis.get("contributions"), 3),
        "至少三步方法": _method_steps_ok(analysis.get("methodSteps")),
        "至少三项实验依据": _useful_list(analysis.get("evidence"), 3),
        # Evidence and paper limitations naturally contain boundary phrases such
        # as “当前材料未说明”; those are useful content, not placeholders.
        "至少三项局限": _valid_text_list(analysis.get("limitations"), 3),
        "综合判断": _useful_text(synthesis.get("thesis"), 30),
        "至少三项研究空白": _valid_text_list(synthesis.get("gaps"), 3),
    }
    categories["structure"] = _category("结构完整性", 25, structure_checks)

    deep = analysis.get("deepDive") if isinstance(analysis.get("deepDive"), dict) else {}
    depth_checks = {
        "关键概念": _deep_items_ok(deep.get("keyConcepts"), 3),
        "技术机制": _useful_list(deep.get("technicalDetails"), 3),
        "实验设置": _useful_list(deep.get("experimentalSetup"), 3),
        "主要发现": _useful_list(deep.get("findings"), 3),
        "精读指南": _useful_list(deep.get("readingGuide"), 3),
    }
    categories["depth"] = _category("深度解读", 20, depth_checks)

    content = _report_text(report, include_related=False)
    anchor_checks = {}
    anchor_details = []
    for anchor in benchmark.get("goldAnchors", []):
        patterns = [str(item) for item in anchor.get("patterns", [])]
        matched = [bool(re.search(pattern, content, re.IGNORECASE)) for pattern in patterns]
        passed = bool(matched) and all(matched)
        label = str(anchor.get("label") or anchor.get("id") or "未命名锚点")
        anchor_checks[label] = passed
        anchor_details.append({
            "id": anchor.get("id"),
            "label": label,
            "matchedPatterns": sum(matched),
            "totalPatterns": len(patterns),
            "passed": passed,
        })
    categories["anchors"] = _category("事实锚点", 25, anchor_checks)
    categories["anchors"]["details"] = anchor_details

    related = [
        item for item in report.get("relatedPapers", [])
        if isinstance(item, dict) and item.get("id") != paper.get("id")
    ]
    explained = [item for item in related if _useful_text(item.get("summary"), 30)]
    compared = [item for item in related if _useful_text(item.get("difference"), 15)]
    bounded = [item for item in related if _useful_text(item.get("evidenceBoundary"), 12)]
    relation_types = {str(item.get("relation")) for item in related if item.get("relation")}
    related_titles = [_normalized(item.get("title")) for item in related if _normalized(item.get("title"))]
    related_checks = {
        "至少五篇相关工作": len(related) >= 5,
        "至少五篇有实质简介": len(explained) >= 5,
        "至少五篇有关键差异": len(compared) >= 5,
        "至少五篇有证据边界": len(bounded) >= 5,
        "关系类型不单一": len(relation_types) >= 2,
        "相关工作题名无重复": len(related_titles) == len(set(related_titles)),
    }
    categories["related"] = _category("相关工作", 15, related_checks)
    categories["related"]["counts"] = {
        "papers": len(related),
        "explained": len(explained),
        "compared": len(compared),
        "bounded": len(bounded),
        "relationTypes": len(relation_types),
        "duplicateTitles": len(related_titles) - len(set(related_titles)),
    }

    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    openalex_meta = meta.get("openAlex") if isinstance(meta.get("openAlex"), dict) else {}
    benchmark_meta = report.get("_benchmark") if isinstance(report.get("_benchmark"), dict) else {}
    provenance = [item for item in report.get("provenance", []) if isinstance(item, dict)]
    trace_checks = {
        "模型调用凭证": bool(meta.get("modelCall", {}).get("verified")),
        "至少两个来源记录": len(provenance) >= 2,
        "原始论文来源": any(str(item.get("url") or "").startswith("http") for item in provenance),
    }
    categories["traceability"] = _category("可追溯性", 5, trace_checks)

    total = round(sum(item["score"] for item in categories.values()), 2)
    return {
        "benchmarkId": benchmark.get("id"),
        "evaluatorVersion": benchmark.get("evaluatorVersion", "1.0.0"),
        "reportVersion": meta.get("version"),
        "generatedAt": meta.get("generatedAt"),
        "score": total,
        "maxScore": 100,
        "categories": categories,
        "runtime": {
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "analysisDepth": meta.get("analysisDepth"),
            "materialScope": meta.get("materialScope"),
            "usage": meta.get("usage") or {},
            "modelCallCount": meta.get("modelCall", {}).get("callCount"),
            "stageCount": meta.get("modelCall", {}).get("stageCount"),
            "validationRetries": meta.get("modelCall", {}).get("validationRetries"),
            "schemaFallbackCount": meta.get("modelCall", {}).get("schemaFallbackCount"),
            "schemaFallbacks": meta.get("modelCall", {}).get("schemaFallbacks") or [],
            "elapsedSeconds": meta.get("elapsedSeconds"),
            "serverElapsedSeconds": meta.get("elapsedSeconds"),
            "modelElapsedSeconds": meta.get("modelCall", {}).get("modelElapsedSeconds"),
            "openAlexCacheHits": openalex_meta.get("cacheHits"),
            "openAlexCacheMisses": openalex_meta.get("cacheMisses"),
            "benchmarkSequence": benchmark_meta.get("pairSequence"),
        },
    }


def compare_scores(before: dict, after: dict) -> dict:
    categories = {}
    keys = sorted(set(before.get("categories", {})) | set(after.get("categories", {})))
    for key in keys:
        old = float(before.get("categories", {}).get(key, {}).get("score", 0))
        new = float(after.get("categories", {}).get(key, {}).get("score", 0))
        categories[key] = {"before": old, "after": new, "delta": round(new - old, 2)}
    old_total = float(before.get("score", 0))
    new_total = float(after.get("score", 0))
    return {
        "benchmarkId": after.get("benchmarkId") or before.get("benchmarkId"),
        "before": old_total,
        "after": new_total,
        "delta": round(new_total - old_total, 2),
        "categories": categories,
        "runtime": {"before": before.get("runtime", {}), "after": after.get("runtime", {})},
    }


def markdown_score(score: dict) -> str:
    lines = [
        f"# PaperScope 基准评分 · {score.get('benchmarkId', '')}",
        "",
        f"总分：**{score.get('score', 0)} / {score.get('maxScore', 100)}**",
        "",
        "| 维度 | 得分 | 满分 | 通过项 |",
        "|---|---:|---:|---:|",
    ]
    for item in score.get("categories", {}).values():
        lines.append(f"| {item['label']} | {item['score']} | {item['maxScore']} | {item['passed']}/{item['total']} |")
    lines.extend(["", "## 未通过检查", ""])
    failed = []
    for item in score.get("categories", {}).values():
        failed.extend(f"- {item['label']}：{label}" for label, ok in item.get("checks", {}).items() if not ok)
    lines.extend(failed or ["- 无"])
    lines.extend(["", "## 运行信息", "", "```json", json.dumps(score.get("runtime", {}), ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def markdown_comparison(comparison: dict) -> str:
    lines = [
        f"# PaperScope 前后对比 · {comparison.get('benchmarkId', '')}",
        "",
        f"总分：**{comparison.get('before', 0)} → {comparison.get('after', 0)}**（{comparison.get('delta', 0):+g}）",
        "",
        "| 维度 | 优化前 | 优化后 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for key, item in comparison.get("categories", {}).items():
        lines.append(f"| {key} | {item['before']} | {item['after']} | {item['delta']:+g} |")
    before_runtime = comparison.get("runtime", {}).get("before", {})
    after_runtime = comparison.get("runtime", {}).get("after", {})
    lines.extend(["", "## 运行成本与材料", "", "| 指标 | 优化前 | 优化后 | 变化 |", "|---|---:|---:|---:|"])
    for label, path in (
        ("总 Token", ("usage", "total_tokens")),
        ("输入 Token", ("usage", "prompt_tokens")),
        ("输出 Token", ("usage", "completion_tokens")),
        ("Prompt Cache 命中 Token", ("usage", "prompt_cache_hit_tokens")),
        ("Prompt Cache 未命中 Token", ("usage", "prompt_cache_miss_tokens")),
        ("推理 Token", ("usage", "reasoning_tokens")),
        ("端到端耗时/秒", ("elapsedSeconds",)),
        ("材料准备/秒", ("materialPreparationSeconds",)),
        ("研究请求/秒", ("researchRequestSeconds",)),
        ("服务端耗时/秒", ("serverElapsedSeconds",)),
        ("模型阶段/秒", ("modelElapsedSeconds",)),
        ("模型调用记录", ("modelCallCount",)),
        ("模型阶段数", ("stageCount",)),
        ("质量修复次数", ("validationRetries",)),
        ("结构降级字段", ("schemaFallbackCount",)),
    ):
        old = _nested_number(before_runtime, path)
        new = _nested_number(after_runtime, path)
        delta = round(new - old, 3) if old is not None and new is not None else None
        lines.append(f"| {label} | {_display_metric(old)} | {_display_metric(new)} | {_display_delta(delta)} |")
    lines.extend([
        "",
        f"- 优化前材料：{before_runtime.get('materialScope') or '—'}",
        f"- 优化后材料：{after_runtime.get('materialScope') or '—'}",
        f"- 执行顺序：摘要版第 {_display_metric(_nested_number(before_runtime, ('benchmarkSequence',)))} 次；全文版第 {_display_metric(_nested_number(after_runtime, ('benchmarkSequence',)))} 次。",
        f"- OpenAlex 缓存：摘要命中/未命中 {_display_metric(_nested_number(before_runtime, ('openAlexCacheHits',)))}/{_display_metric(_nested_number(before_runtime, ('openAlexCacheMisses',)))}；全文命中/未命中 {_display_metric(_nested_number(after_runtime, ('openAlexCacheHits',)))}/{_display_metric(_nested_number(after_runtime, ('openAlexCacheMisses',)))}。",
        "- 耗时比较会受到执行顺序、OpenAlex 缓存和模型 Prompt Cache 影响；需要反向顺序或重复运行后再判断稳定差异。",
        "",
    ])
    return "\n".join(lines)


def _nested_number(payload: dict, path: tuple[str, ...]) -> float | None:
    value = payload
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _display_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:g}"


def _display_delta(value: float | None) -> str:
    return "—" if value is None else f"{value:+g}"


def _category(label: str, max_score: float, checks: dict[str, bool]) -> dict:
    total = len(checks)
    passed = sum(bool(value) for value in checks.values())
    score = round(max_score * passed / total, 2) if total else 0.0
    return {"label": label, "score": score, "maxScore": max_score, "passed": passed, "total": total, "checks": checks}


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _useful_text(value: object, minimum: int) -> bool:
    text = str(value or "").strip()
    return len(text) >= minimum and not any(re.search(pattern, text, re.IGNORECASE) for pattern in PLACEHOLDER_PATTERNS)


def _useful_list(value: object, minimum: int) -> bool:
    return isinstance(value, list) and sum(_useful_text(item, 8) for item in value if isinstance(item, str)) >= minimum


def _valid_text_list(value: object, minimum: int) -> bool:
    return isinstance(value, list) and sum(len(item.strip()) >= 8 for item in value if isinstance(item, str)) >= minimum


def _method_steps_ok(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return sum(
        _useful_text(item.get("label"), 1) and _useful_text(item.get("text"), 8)
        for item in value if isinstance(item, dict)
    ) >= 3


def _deep_items_ok(value: object, minimum: int) -> bool:
    if not isinstance(value, list):
        return False
    return sum(
        _useful_text(item.get("term"), 2) and _useful_text(item.get("explanation"), 10)
        for item in value if isinstance(item, dict)
    ) >= minimum


def _report_text(report: dict, include_related: bool) -> str:
    selected = {
        "paper": report.get("paper", {}),
        "analysis": report.get("analysis", {}),
        "synthesis": report.get("synthesis", {}),
    }
    if include_related:
        selected["relatedPapers"] = report.get("relatedPapers", [])
    return json.dumps(selected, ensure_ascii=False)
