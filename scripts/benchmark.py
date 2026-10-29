#!/usr/bin/env python3
"""Capture, score and compare reproducible PaperScope benchmark runs."""

from __future__ import annotations

import argparse
import base64
import json
import re
from statistics import mean, median
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation import (  # noqa: E402
    compare_scores,
    load_json,
    markdown_comparison,
    markdown_score,
    score_report,
)


class BenchmarkRequestError(RuntimeError):
    """A local PaperScope request failed and should be recorded as a run."""


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperScope 固定论文质量评测")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="调用本地 PaperScope 并保存一次运行")
    capture.add_argument("--benchmark", default=str(PROJECT_ROOT / "evaluation/benchmarks/reasc.json"))
    capture.add_argument("--api-url", default="http://127.0.0.1:8765")
    capture.add_argument("--label", required=True, help="例如 v0.5.1-baseline")
    capture.add_argument("--source", help="可选 PDF/TXT/MD；PDF 会交给本地提取接口")
    capture.add_argument("--material", choices=("abstract", "auto-oa", "source"), help="材料模式；默认有 --source 时为 source，否则为 abstract")

    repeat = subparsers.add_parser("repeat", help="连续捕获多次并汇总稳定性")
    repeat.add_argument("--benchmark", default=str(PROJECT_ROOT / "evaluation/benchmarks/reasc.json"))
    repeat.add_argument("--api-url", default="http://127.0.0.1:8765")
    repeat.add_argument("--label", required=True, help="例如 v0.6.0-staged")
    repeat.add_argument("--runs", type=int, default=3)
    repeat.add_argument("--source", help="可选 PDF/TXT/MD；PDF 会交给本地提取接口")
    repeat.add_argument("--material", choices=("abstract", "auto-oa", "source"), help="材料模式；默认有 --source 时为 source，否则为 abstract")

    pair = subparsers.add_parser("pair", help="成对捕获摘要版与 OpenAlex 自动开放全文版")
    pair.add_argument("--benchmark", default=str(PROJECT_ROOT / "evaluation/benchmarks/reasc.json"))
    pair.add_argument("--api-url", default="http://127.0.0.1:8765")
    pair.add_argument("--label", required=True, help="例如 v0.7.4-deepseek")
    pair.add_argument(
        "--order",
        choices=("abstract-first", "fulltext-first"),
        default="abstract-first",
        help="执行顺序；反向再跑一次可识别冷/热缓存偏差",
    )

    score = subparsers.add_parser("score", help="对已有报告打分")
    score.add_argument("report")
    score.add_argument("--benchmark", default=str(PROJECT_ROOT / "evaluation/benchmarks/reasc.json"))
    score.add_argument("--output")

    compare = subparsers.add_parser("compare", help="比较两份评分或报告")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--benchmark", default=str(PROJECT_ROOT / "evaluation/benchmarks/reasc.json"))
    compare.add_argument("--output")

    args = parser.parse_args()
    if args.command == "capture":
        _capture(args)
    elif args.command == "repeat":
        _repeat(args)
    elif args.command == "pair":
        _pair(args)
    elif args.command == "score":
        _score(args)
    else:
        _compare(args)


def _capture(args: argparse.Namespace, exit_on_failure: bool = True) -> Path:
    benchmark = load_json(args.benchmark)
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / benchmark["id"] / _safe_label(args.label)
    run_dir.mkdir(parents=True, exist_ok=True)
    source_text = ""
    source_info = None
    material_mode = _resolve_material_mode(getattr(args, "material", None), getattr(args, "source", None))
    started = datetime.now(timezone.utc)
    prepared_at = None
    research_started = None
    try:
        source_text, source_info = _prepare_material(
            args.api_url,
            benchmark,
            material_mode,
            getattr(args, "source", None),
        )
        prepared_at = datetime.now(timezone.utc)
        payload = {
            "query": benchmark["query"],
            "demo": False,
            "depth": benchmark.get("depth", "deep"),
            "sourceText": source_text,
            "sourceInfo": source_info,
        }
        research_started = datetime.now(timezone.utc)
        report = _request_json(f"{args.api_url.rstrip('/')}/api/research", payload)
    except BenchmarkRequestError as error:
        finished = datetime.now(timezone.utc)
        failure = {
            "benchmarkId": benchmark["id"],
            "label": args.label,
            "status": "failed",
            "capturedAt": finished.isoformat(),
            "elapsedSeconds": round((finished - started).total_seconds(), 3),
            "materialPreparationSeconds": round((prepared_at - started).total_seconds(), 3) if prepared_at else None,
            "researchRequestSeconds": round((finished - research_started).total_seconds(), 3) if research_started else None,
            "sourceFile": Path(args.source).name if args.source else None,
            "sourceCharacters": len(source_text),
            "materialMode": material_mode,
            "pairSequence": getattr(args, "pair_sequence", None),
            "error": str(error),
        }
        result = score_report({}, benchmark)
        result.update({"status": "failed", "error": str(error)})
        result["runtime"]["elapsedSeconds"] = failure["elapsedSeconds"]
        result["runtime"]["materialPreparationSeconds"] = failure["materialPreparationSeconds"]
        result["runtime"]["researchRequestSeconds"] = failure["researchRequestSeconds"]
        result["runtime"]["benchmarkSequence"] = failure["pairSequence"]
        (run_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "score.md").write_text(markdown_score(result), encoding="utf-8")
        print(f"运行失败，已记录：{run_dir / 'failure.json'}")
        print(f"错误：{error}")
        if exit_on_failure:
            raise SystemExit(2) from error
        return run_dir
    finished = datetime.now(timezone.utc)
    report["_benchmark"] = {
        "id": benchmark["id"],
        "label": args.label,
        "capturedAt": finished.isoformat(),
        "elapsedSeconds": round((finished - started).total_seconds(), 3),
        "materialPreparationSeconds": round((prepared_at - started).total_seconds(), 3),
        "researchRequestSeconds": round((finished - research_started).total_seconds(), 3),
        "sourceFile": Path(args.source).name if args.source else None,
        "sourceCharacters": len(source_text),
        "materialMode": material_mode,
        "pairSequence": getattr(args, "pair_sequence", None),
    }
    report_path = run_dir / "report.json"
    score_path = run_dir / "score.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    result = score_report(report, benchmark)
    result["runtime"]["elapsedSeconds"] = report["_benchmark"]["elapsedSeconds"]
    result["runtime"]["materialPreparationSeconds"] = report["_benchmark"]["materialPreparationSeconds"]
    result["runtime"]["researchRequestSeconds"] = report["_benchmark"]["researchRequestSeconds"]
    score_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "score.md").write_text(markdown_score(result), encoding="utf-8")
    print(f"报告：{report_path}")
    print(f"评分：{result['score']} / 100（{score_path}）")
    return run_dir


def _repeat(args: argparse.Namespace) -> None:
    if not 2 <= args.runs <= 10:
        raise SystemExit("runs 必须在 2—10 之间。")
    run_dirs = []
    for index in range(1, args.runs + 1):
        run_args = argparse.Namespace(
            benchmark=args.benchmark,
            api_url=args.api_url,
            label=f"{args.label}-run-{index}",
            source=args.source,
            material=getattr(args, "material", None),
        )
        print(f"\n[{index}/{args.runs}] {run_args.label}")
        run_dirs.append(_capture(run_args, exit_on_failure=False))
    scores = [load_json(path / "score.json") for path in run_dirs]
    summary = _summarize_scores(scores, args.label)
    output_dir = PROJECT_ROOT / "evaluation" / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{_safe_label(args.label)}.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown_summary(summary), encoding="utf-8")
    print(f"\n成功率：{summary['successfulRuns']}/{summary['totalRuns']}（{summary['successRatePercent']}%）")
    print(f"汇总：{output}")


def _pair(args: argparse.Namespace) -> None:
    run_dirs = {}
    for sequence, (suffix, material) in enumerate(_pair_materials(args.order), start=1):
        print(f"\n[{suffix}] {args.label}-{suffix}")
        run_dirs[suffix] = _capture(argparse.Namespace(
            benchmark=args.benchmark,
            api_url=args.api_url,
            label=f"{args.label}-{suffix}",
            source=None,
            material=material,
            pair_sequence=sequence,
        ), exit_on_failure=False)
    benchmark = load_json(args.benchmark)
    before = load_json(run_dirs["abstract"] / "score.json")
    after = load_json(run_dirs["fulltext"] / "score.json")
    comparison = compare_scores(before, after)
    output = PROJECT_ROOT / "evaluation" / "comparisons" / f"{_safe_label(args.label)}-abstract-vs-fulltext.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_comparison(comparison), encoding="utf-8")
    print(f"\n摘要版 → 自动全文版：{comparison['before']} → {comparison['after']}（{comparison['delta']:+g}）")
    print(f"对比：{output}")


def _pair_materials(order: str) -> tuple[tuple[str, str], tuple[str, str]]:
    pairs = (("abstract", "abstract"), ("fulltext", "auto-oa"))
    return tuple(reversed(pairs)) if order == "fulltext-first" else pairs


def _score(args: argparse.Namespace) -> None:
    report = load_json(args.report)
    benchmark = load_json(args.benchmark)
    result = score_report(report, benchmark)
    output = Path(args.output) if args.output else Path(args.report).with_name("score.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_score(result), encoding="utf-8")
    print(f"评分：{result['score']} / 100（{output}）")


def _compare(args: argparse.Namespace) -> None:
    benchmark = load_json(args.benchmark)
    before = _as_score(load_json(args.before), benchmark)
    after = _as_score(load_json(args.after), benchmark)
    result = compare_scores(before, after)
    output = Path(args.output) if args.output else PROJECT_ROOT / "evaluation" / "latest-comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_comparison(result), encoding="utf-8")
    print(f"总分：{result['before']} → {result['after']}（{result['delta']:+g}）")
    print(f"对比：{output}")


def _as_score(payload: dict, benchmark: dict) -> dict:
    return payload if "maxScore" in payload and "categories" in payload else score_report(payload, benchmark)


def _load_source(api_url: str, source: str) -> str:
    path = Path(source)
    if not path.is_file():
        raise SystemExit(f"材料文件不存在：{path}")
    if path.suffix.lower() == ".pdf":
        payload = {"data": base64.b64encode(path.read_bytes()).decode("ascii")}
        extracted = _request_json(f"{api_url.rstrip('/')}/api/extract/pdf", payload)
        text = str(extracted.get("text") or "")
    else:
        text = path.read_text(encoding="utf-8")
    return text[:80_000]


def _resolve_material_mode(material: str | None, source: str | None) -> str:
    mode = material or ("source" if source else "abstract")
    if mode == "source" and not source:
        raise SystemExit("material=source 时必须提供 --source。")
    if mode != "source" and source:
        raise SystemExit("--source 只能与 material=source 一起使用。")
    return mode


def _prepare_material(api_url: str, benchmark: dict, mode: str, source: str | None) -> tuple[str, dict | None]:
    if mode == "source":
        return _load_source(api_url, str(source)), {"kind": "user-file", "label": Path(str(source)).name}
    if mode == "auto-oa":
        result = _request_json(
            f"{api_url.rstrip('/')}/api/fulltext/open-access",
            {"query": benchmark["query"]},
        )
        text = str(result.get("text") or "")[:80_000]
        source_info = result.get("source") if isinstance(result.get("source"), dict) else None
        if not text or source_info is None:
            raise BenchmarkRequestError("PaperScope 未返回可用的开放全文与来源信息。")
        return text, source_info
    return "", {
        "kind": "abstract-fallback",
        "label": "ReASC 基准主动使用公开摘要",
        "reason": "benchmark-abstract-mode",
    }


def _request_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=240) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise BenchmarkRequestError(f"PaperScope 返回 HTTP {error.code}：{body}") from error
    except URLError as error:
        raise BenchmarkRequestError(f"无法连接本地 PaperScope：{error.reason}") from error
    if not isinstance(result, dict):
        raise BenchmarkRequestError("PaperScope 返回了无效 JSON。")
    return result


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.")
    if not cleaned:
        raise SystemExit("label 不能为空。")
    return cleaned[:80]


def _summarize_scores(scores: list[dict], label: str) -> dict:
    successful = [item for item in scores if item.get("status") != "failed"]

    def values(path: tuple[str, ...]) -> list[float]:
        result = []
        for item in successful:
            current = item
            for key in path:
                current = current.get(key, {}) if isinstance(current, dict) else {}
            if isinstance(current, (int, float)):
                result.append(float(current))
        return result

    def stats(items: list[float]) -> dict:
        if not items:
            return {"min": None, "median": None, "mean": None, "max": None}
        return {
            "min": round(min(items), 2),
            "median": round(median(items), 2),
            "mean": round(mean(items), 2),
            "max": round(max(items), 2),
        }

    total = len(scores)
    return {
        "label": label,
        "benchmarkId": next((item.get("benchmarkId") for item in scores if item.get("benchmarkId")), None),
        "evaluatorVersion": next((item.get("evaluatorVersion") for item in scores if item.get("evaluatorVersion")), None),
        "totalRuns": total,
        "successfulRuns": len(successful),
        "failedRuns": total - len(successful),
        "successRatePercent": round(100 * len(successful) / total, 2) if total else 0,
        "score": stats(values(("score",))),
        "totalTokens": stats(values(("runtime", "usage", "total_tokens"))),
        "promptCacheHitTokens": stats(values(("runtime", "usage", "prompt_cache_hit_tokens"))),
        "promptCacheMissTokens": stats(values(("runtime", "usage", "prompt_cache_miss_tokens"))),
        "reasoningTokens": stats(values(("runtime", "usage", "reasoning_tokens"))),
        "elapsedSeconds": stats(values(("runtime", "elapsedSeconds"))),
        "materialPreparationSeconds": stats(values(("runtime", "materialPreparationSeconds"))),
        "researchRequestSeconds": stats(values(("runtime", "researchRequestSeconds"))),
        "serverElapsedSeconds": stats(values(("runtime", "serverElapsedSeconds"))),
        "modelElapsedSeconds": stats(values(("runtime", "modelElapsedSeconds"))),
        "modelCallCount": stats(values(("runtime", "modelCallCount"))),
        "validationRetries": stats(values(("runtime", "validationRetries"))),
        "schemaFallbackCount": stats(values(("runtime", "schemaFallbackCount"))),
        "failures": [item.get("error") for item in scores if item.get("status") == "failed"],
    }


def _markdown_summary(summary: dict) -> str:
    lines = [
        f"# PaperScope 重复盲测汇总 · {summary.get('label', '')}",
        "",
        f"成功率：**{summary.get('successfulRuns', 0)}/{summary.get('totalRuns', 0)}（{summary.get('successRatePercent', 0)}%）**",
        "",
        "| 指标 | 最低 | 中位数 | 平均 | 最高 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("score", "评分"),
        ("totalTokens", "总 Token"),
        ("promptCacheHitTokens", "Prompt Cache 命中 Token"),
        ("promptCacheMissTokens", "Prompt Cache 未命中 Token"),
        ("reasoningTokens", "推理 Token"),
        ("elapsedSeconds", "耗时/秒"),
        ("materialPreparationSeconds", "材料准备/秒"),
        ("researchRequestSeconds", "研究请求/秒"),
        ("serverElapsedSeconds", "服务端/秒"),
        ("modelElapsedSeconds", "模型阶段/秒"),
        ("modelCallCount", "模型调用次数"),
        ("validationRetries", "校验重试次数"),
        ("schemaFallbackCount", "结构降级数量"),
    ):
        item = summary.get(key, {})
        lines.append(f"| {label} | {item.get('min')} | {item.get('median')} | {item.get('mean')} | {item.get('max')} |")
    if summary.get("failures"):
        lines.extend(["", "## 失败记录", ""])
        lines.extend(f"- {error}" for error in summary["failures"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
