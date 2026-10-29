"""Research workflow orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Callable

from .config import Settings, update_runtime_openalex_status
from .integrations import IntegrationNotConfigured, ProviderRequestError, build_provider_bundle
from .mock_data import build_demo_report
from .pdf_text import download_open_access_pdf, extract_pdf_text


PIPELINE = [
    {
        "id": "resolve",
        "label": "识别种子论文",
        "description": "解析标题、DOI、arXiv 或论文链接，确定唯一文献。",
    },
    {
        "id": "read",
        "label": "读取论文信息",
        "description": "获取摘要、元数据和可追踪的原始来源。",
    },
    {
        "id": "discover",
        "label": "主动发现相关工作",
        "description": "组合引用、语义检索和方法关键词扩展候选文献。",
    },
    {
        "id": "verify",
        "label": "去重与证据核验",
        "description": "合并版本、检查来源，并区分事实与模型分析。",
    },
    {
        "id": "analyze",
        "label": "结构化解读",
        "description": "抽取问题、方法、贡献、实验依据与局限。",
    },
    {
        "id": "synthesize",
        "label": "整理与可视化",
        "description": "生成时间线、关系图、研究簇、空白和阅读顺序。",
    },
]


def get_pipeline() -> list[dict]:
    return [dict(step) for step in PIPELINE]


def run_research(
    query: str,
    demo: bool,
    settings: Settings,
    source_text: str = "",
    depth: str = "standard",
    source_info: dict | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> dict:
    clean_query = " ".join(query.split())
    if not clean_query:
        raise ValueError("请输入论文标题、DOI、arXiv 编号或论文链接。")
    if len(clean_query) > 1000:
        raise ValueError("输入内容过长，请提供单篇论文的标题或标识符。")
    clean_source = source_text.strip()
    if len(clean_source) > 80_000:
        raise ValueError("全文材料最多支持 80,000 个字符，请保留摘要、方法和实验等关键章节。")
    clean_depth = depth.strip().lower()
    if clean_depth not in {"standard", "deep"}:
        raise ValueError("分析深度必须是 standard 或 deep。")

    if demo:
        _report_progress(progress, "resolve", "正在载入演示论文")
        report = build_demo_report(clean_query)
        report["meta"]["generatedAt"] = datetime.now(timezone.utc).isoformat()
        report["meta"]["analysisDepth"] = clean_depth
        report["meta"]["materialScope"] = "内置演示材料"
        report["pipeline"] = get_pipeline()
        _report_progress(progress, "synthesize", "演示报告已整理完成")
        return report

    providers = build_provider_bundle(settings)
    if not settings.llm_ready or providers.llm is None:
        raise IntegrationNotConfigured(
            "模型增强研究尚未启用。请先在后端配置支持的模型 API；当前可使用演示模式。"
        )

    if not settings.paper_search_provider:
        if not _is_echosight_demo_query(clean_query):
            raise IntegrationNotConfigured(
                "模型 API 已接入，但论文检索 API 尚未配置。当前模型测试仅支持默认的 EchoSight 演示论文。"
            )
        _report_progress(progress, "resolve", "正在识别演示论文")
        report = build_demo_report(clean_query)
        _report_progress(progress, "analyze", "模型正在解读论文")
        patch = providers.llm.enhance_report(report)
        provider = patch.pop("_provider")
        related_notes = patch.pop("relatedNotes", {})
        report["paper"]["oneSentence"] = patch["oneSentence"]
        report["analysis"].update({
            "problem": patch["problem"],
            "contributions": patch["contributions"],
            "methodSteps": patch["methodSteps"],
            "evidence": patch["evidence"],
            "limitations": patch["limitations"],
            "deepDive": patch.get("deepDive", {}),
        })
        report["synthesis"].update({"thesis": patch["thesis"], "gaps": patch["gaps"]})
        _merge_related_notes(report, related_notes)
        report["meta"].update({
            "mode": "llm-assisted-demo",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "model": provider["model"],
            "provider": provider["name"],
            "usage": provider["usage"],
            "modelCall": _model_call_meta(provider),
            "notice": (
                f"{provider['label']} · {provider['model']} 已实时生成论文解读与综合判断；"
                "论文元数据和相关工作仍来自内置 EchoSight 演示资料，尚未执行真实论文检索。"
            ),
        })
        report["provenance"].insert(0, {
            "label": f"{provider['label']} · {provider['model']}",
            "kind": "模型实时生成",
            "url": "",
            "status": "基于内置材料",
        })
        report["pipeline"] = get_pipeline()
        _report_progress(progress, "synthesize", "正在整理模型分析结果")
        return report

    if not settings.live_ready or not providers.ready:
        raise IntegrationNotConfigured("实时研究尚未启用。请检查模型和论文数据源配置。")

    _report_progress(progress, "resolve", "正在解析标题、DOI 或论文链接")
    seed = providers.paper_search.resolve_paper(clean_query)
    _report_progress(progress, "read", "已定位种子论文，正在读取元数据和摘要")
    _report_progress(progress, "discover", "正在检索引用、被引与语义相关论文")
    related = providers.paper_search.search_related(seed, [seed["title"]])
    citations = providers.citation_graph.expand_citations(seed) if providers.citation_graph else []
    _report_progress(progress, "verify", "正在合并版本、去重并核对来源")
    report = _build_live_report(clean_query, seed, citations + related, clean_source, clean_depth, source_info)
    rate_status_getter = getattr(providers.paper_search, "rate_limit_status", None)
    openalex_status = rate_status_getter() if callable(rate_status_getter) else {}
    report["meta"]["openAlex"] = openalex_status
    update_runtime_openalex_status(openalex_status)
    if openalex_status.get("mode") == "api-key":
        for item in report.get("provenance", []):
            if item.get("kind") == "OpenAlex 公共 API":
                item["status"] = "免费 API Key 模式"
    _report_progress(progress, "analyze", "模型正在生成种子论文结构化解读")
    progress_setter = getattr(providers.llm, "set_progress_callback", None)
    if callable(progress_setter):
        progress_setter(progress)
    patch = providers.llm.enhance_report(report)
    _report_progress(progress, "synthesize", "正在综合相关工作并生成可视化数据")
    provider = patch.pop("_provider")
    related_notes = patch.pop("relatedNotes", {})
    report["paper"]["oneSentence"] = patch["oneSentence"]
    report["analysis"].update({
        "problem": patch["problem"],
        "contributions": patch["contributions"],
        "methodSteps": patch["methodSteps"],
        "evidence": patch["evidence"],
        "limitations": patch["limitations"],
        "deepDive": patch.get("deepDive", {}),
    })
    report["synthesis"].update({"thesis": patch["thesis"], "gaps": patch["gaps"]})
    _merge_related_notes(report, related_notes)
    report.pop("_sourceMaterial", None)
    report["meta"].update({
        "model": provider["model"],
        "provider": provider["name"],
        "usage": provider["usage"],
        "modelCall": _model_call_meta(provider),
        "notice": (
            f"{provider['label']} · {provider['model']} 负责{('全文优先的深度' if clean_depth == 'deep' else '标准')}结构化解读；"
            "OpenAlex 免费模式提供论文元数据、相关工作与真实引用关系，Crossref 仅作 DOI 元数据回退。"
            + ("本次未获得全文，详细模式已明确降级为摘要材料分析。" if report["meta"].get("materialQuality", {}).get("degraded") else "")
        ),
    })
    report["provenance"].insert(0, {
        "label": f"{provider['label']} · {provider['model']}",
        "kind": "模型结构化分析",
        "url": "",
        "status": "依据用户材料与已检索材料生成" if clean_source else "仅依据已检索材料生成",
    })
    return report


def fetch_open_access_fulltext(query: str, settings: Settings) -> dict:
    """Resolve one paper and extract an OpenAlex-listed OA PDF without persisting it."""
    clean_query = " ".join(query.split())
    if not clean_query:
        raise ValueError("请先输入论文标题、DOI、arXiv 编号或论文链接。")
    providers = build_provider_bundle(settings)
    if providers.paper_search is None:
        raise IntegrationNotConfigured("论文数据源尚未配置。")
    seed = providers.paper_search.resolve_paper(clean_query)
    pdf_url = str(seed.get("openAccessPdfUrl") or "").strip()
    if not pdf_url:
        raise ValueError("OpenAlex 未收录这篇论文的可直接读取开放 PDF。")
    downloaded = download_open_access_pdf(pdf_url)
    extracted = extract_pdf_text(downloaded["bytes"])
    identity = _verify_fulltext_identity(extracted["text"], seed)
    if not identity["matched"]:
        raise ValueError("开放 PDF 的题名或 DOI 与种子论文不匹配，已停止使用。")
    rate_status_getter = getattr(providers.paper_search, "rate_limit_status", None)
    openalex_status = rate_status_getter() if callable(rate_status_getter) else {}
    update_runtime_openalex_status(openalex_status)
    source = {
        "kind": "open-access-pdf",
        "label": seed.get("title") or "开放全文 PDF",
        "url": downloaded["url"],
        "landingUrl": seed.get("openAccessLandingUrl") or seed.get("sourceUrl") or "",
        "version": seed.get("openAccessVersion") or "open-access",
        "doi": seed.get("doi") or "",
        "identity": identity,
    }
    return {"ok": True, **extracted, "source": source, "openAlex": openalex_status}


def _report_progress(progress: Callable[[str, str], None] | None, step_id: str, detail: str) -> None:
    if progress is not None:
        progress(step_id, detail)


def test_llm_connection(settings: Settings) -> dict:
    providers = build_provider_bundle(settings)
    if not settings.llm_ready or providers.llm is None:
        raise IntegrationNotConfigured("模型 API 尚未配置。")
    return providers.llm.test_connection()


def _model_call_meta(provider: dict) -> dict:
    calls = provider.get("calls") if isinstance(provider.get("calls"), list) else []
    return {
        "verified": bool(
            provider.get("responseId")
            or provider.get("requestId")
            or provider.get("usage")
            or provider.get("cacheHits")
        ),
        "provider": provider.get("name"),
        "providerLabel": provider.get("label"),
        "model": provider.get("model"),
        "responseId": provider.get("responseId"),
        "requestId": provider.get("requestId"),
        "providerCreatedAt": provider.get("providerCreatedAt"),
        "usage": provider.get("usage") or {},
        "callCount": len(calls) or 1,
        "validationRetries": int(provider.get("validationRetries") or 0),
        "stageCount": int(provider.get("stageCount") or 1),
        "schemaFallbackCount": int(provider.get("schemaFallbackCount") or 0),
        "schemaFallbacks": provider.get("schemaFallbacks") or [],
        "cacheHits": int(provider.get("cacheHits") or 0),
        "resumedStages": provider.get("resumedStages") or [],
        "transportRetries": int(provider.get("transportRetries") or 0),
        "modelElapsedSeconds": float(provider.get("modelElapsedSeconds") or 0),
        "calls": calls,
    }


def test_data_connection(settings: Settings) -> dict:
    providers = build_provider_bundle(settings)
    if providers.paper_search is None:
        raise IntegrationNotConfigured("论文数据源尚未配置。")
    tester = getattr(providers.paper_search, "test_connection", None)
    if tester is None:
        raise IntegrationNotConfigured("当前论文数据源不支持连接测试。")
    return tester()


def _build_live_report(
    seed_query: str,
    seed: dict,
    candidates: list[dict],
    source_text: str = "",
    depth: str = "standard",
    source_info: dict | None = None,
) -> dict:
    clean_source_info = _clean_source_info(source_info)
    material_scope = _material_scope(source_text, clean_source_info)
    related = _deduplicate_papers(seed, candidates, limit=16)
    seed_item = _display_paper(seed, "种子论文", "本次研究的中心论文。")
    papers = [seed_item] + [_display_paper(item, item.get("relation", "语义相关"), item.get("why", "由 OpenAlex 发现。")) for item in related]
    relations = []
    for item in papers[1:]:
        if item["relation"] == "参考文献":
            relations.append({"from": item["id"], "to": seed_item["id"], "type": "真实引用", "label": "种子论文引用该工作"})
        elif item["relation"] == "后续引用":
            relations.append({"from": seed_item["id"], "to": item["id"], "type": "真实引用", "label": "该工作引用种子论文"})
        elif item["relation"] == "语义检索":
            relations.append({"from": seed_item["id"], "to": item["id"], "type": "语义相关", "label": "OpenAlex 摘要语义检索"})
        else:
            relations.append({"from": seed_item["id"], "to": item["id"], "type": "语义相关", "label": "OpenAlex 相关工作推荐"})

    abstract = seed.get("abstract") or "OpenAlex 当前没有收录该论文摘要；模型只能依据题名、元数据和相关工作进行保守归纳。"
    clusters = []
    for relation, name, insight in (
        ("参考文献", "参考文献回溯", "这些论文由种子论文明确引用，可用于理解其问题来源与方法基础。"),
        ("后续引用", "后续引用扩展", "这些论文明确引用了种子论文，可用于观察后续采用、比较或延伸。"),
        ("语义相关", "语义相关工作", "这些论文来自 OpenAlex 相关工作推荐，不自动等同于真实引用。"),
        ("语义检索", "稀疏结果补充", "这些论文来自摘要语义检索，用于补充引用数据尚未形成的新论文。"),
    ):
        group = [item["shortTitle"] for item in papers if item["relation"] == relation]
        if group:
            clusters.append({"name": name, "papers": group, "insight": insight})
    if not clusters:
        clusters.append({"name": "待扩展", "papers": [seed_item["shortTitle"]], "insight": "当前免费数据源没有返回足够的相关工作。"})

    short_abstract = abstract[:700] + ("…" if len(abstract) > 700 else "")
    year = seed.get("year") or 0
    report = {
        "meta": {
            "mode": "live",
            "seedQuery": seed_query,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "version": "0.7.4",
            "analysisDepth": depth,
            "materialScope": material_scope,
            "materialQuality": _material_quality(source_text, clean_source_info, depth),
            "notice": "正在使用 OpenAlex 免费模式检索，模型将依据这些材料生成结构化解读。",
        },
        "paper": {
            "id": seed_item["id"],
            "title": seed["title"],
            "shortTitle": seed["shortTitle"],
            "authors": seed["authors"],
            "year": year,
            "venue": seed["venue"],
            "identifier": seed["identifier"],
            "sourceUrl": seed["sourceUrl"],
            "oneSentence": "模型正在依据公开摘要生成中文方法定位。",
            "tags": seed.get("tags") or ["OpenAlex", "实时检索"],
        },
        "analysis": {
            "problem": short_abstract,
            "contributions": [
                "待模型从已收录摘要中识别论文的主要贡献。",
                "待模型区分论文事实、作者主张与元数据线索。",
                "待模型结合相关工作说明该论文在研究脉络中的位置。",
            ],
            "methodSteps": [
                {"label": "问题", "text": "依据论文摘要识别研究对象、任务和主要约束。"},
                {"label": "方法", "text": "依据论文摘要整理方法模块及其逻辑关系。"},
                {"label": "验证", "text": "依据摘要中的实验描述总结证据，并标记未核验内容。"},
            ],
            "evidence": [
                f"OpenAlex 收录题名、作者、年份与发表场所；当前引用计数为 {seed.get('citedByCount', 0)}。",
                f"免费检索共发现 {len(papers) - 1} 篇参考文献、后续引用或语义相关工作。",
                "摘要与引用关系来自公开学术元数据；具体实验数字仍应回到论文原文核验。",
            ],
            "limitations": [
                "免费模式不自动下载或解析论文 PDF，分析深度受公开摘要完整度限制。",
                "OpenAlex 的相关工作推荐是语义关系，不代表论文之间存在真实引用。",
                "引用计数与发表信息可能存在更新延迟，关键结论应以论文原文为准。",
            ],
            "deepDive": {
                "keyConcepts": [],
                "technicalDetails": [],
                "experimentalSetup": [],
                "findings": [],
                "readingGuide": [],
            },
        },
        "searchPlan": {
            "queries": [seed["title"], seed.get("doi") or seed.get("openalexId") or seed_query],
            "channels": ["OpenAlex 论文识别", "参考文献回溯", "后续引用扩展", "OpenAlex 相关工作推荐", "稀疏时单次摘要语义检索"],
            "rules": ["优先 DOI 精确匹配", "少于 3 篇时才触发语义回退", "引用与语义相关分开标注", "合并重复版本", "不自动下载 PDF"],
        },
        "relatedPapers": papers,
        "relations": relations,
        "synthesis": {
            "thesis": "结构化综合将在模型读取公开摘要与已核验元数据后生成。",
            "clusters": clusters,
            "gaps": [
                "需要结合论文全文复核方法细节与实验设置。",
                "需要检查公开摘要未覆盖的消融实验和失败案例。",
                "需要确认语义相关工作与种子论文的真实关系。",
                "需要在统一任务、数据与成本约束下比较方法差异。",
            ],
            "readingOrder": [item["shortTitle"] for item in sorted(papers, key=lambda paper: (paper["year"] or 9999, paper["shortTitle"]))],
        },
        "provenance": [
            {"label": seed["title"], "kind": "种子论文元数据", "url": seed["sourceUrl"], "status": f"{seed.get('metadataSource', 'OpenAlex')} 返回"},
            {"label": "相关工作与引用网络", "kind": "OpenAlex 公共 API", "url": "https://openalex.org/", "status": "免费匿名模式"},
            {"label": "DOI 元数据回退", "kind": "Crossref 公共 API", "url": "https://www.crossref.org/", "status": "仅在需要时调用"},
        ],
        "pipeline": get_pipeline(),
        "_sourceMaterial": {
            "abstract": abstract,
            "fullText": source_text,
            "analysisDepth": depth,
            "materialScope": material_scope,
            "citedByCount": seed.get("citedByCount", 0),
        },
    }
    if source_text:
        is_oa = clean_source_info.get("kind") == "open-access-pdf"
        report["provenance"].insert(0, {
            "label": clean_source_info.get("label") or ("开放全文 PDF" if is_oa else "用户提供的论文全文或重点摘录"),
            "kind": "OpenAlex 定位的开放全文" if is_oa else "本次请求材料",
            "url": clean_source_info.get("url") if is_oa else "",
            "status": f"{clean_source_info.get('version') or '开放版本'}；仅在内存中参与分析，不写入项目文件" if is_oa else "仅在内存中参与分析，不写入项目文件",
        })
    elif clean_source_info.get("kind") == "abstract-fallback":
        reason = clean_source_info.get("reason") or "用户明确选择仅使用公开摘要"
        report["provenance"].insert(0, {
            "label": clean_source_info.get("label") or "详细模式材料降级",
            "kind": "材料边界",
            "url": "",
            "status": reason,
        })
    return report


def _clean_source_info(source_info: dict | None) -> dict:
    if not isinstance(source_info, dict):
        return {}
    kind = str(source_info.get("kind") or "").strip()
    if kind not in {"open-access-pdf", "user-file", "abstract-fallback"}:
        return {}
    url = str(source_info.get("url") or "").strip()
    if url and not url.startswith("https://"):
        url = ""
    return {
        "kind": kind,
        "label": str(source_info.get("label") or "").strip()[:500],
        "url": url[:2000],
        "version": str(source_info.get("version") or "").strip()[:100],
        "reason": str(source_info.get("reason") or "").strip()[:500],
    }


def _material_scope(source_text: str, source_info: dict) -> str:
    if not source_text:
        if source_info.get("kind") == "abstract-fallback":
            return "OpenAlex 公开摘要（详细模式材料降级）"
        return "OpenAlex 公开摘要"
    if source_info.get("kind") == "open-access-pdf":
        version = source_info.get("version") or "开放版本"
        return f"OpenAlex 开放 PDF（{version}）"
    return "用户提供全文/摘录"


def _material_quality(source_text: str, source_info: dict, depth: str) -> dict:
    if source_text and source_info.get("kind") == "open-access-pdf":
        return {"level": "full-text", "degraded": False, "label": "已验证开放全文"}
    if source_text:
        return {"level": "user-material", "degraded": False, "label": "用户提供材料"}
    degraded = depth == "deep"
    return {
        "level": "abstract-only",
        "degraded": degraded,
        "label": "详细模式已降级为摘要材料" if degraded else "公开摘要",
        "reason": source_info.get("reason") or ("未提供全文" if degraded else ""),
    }


def _verify_fulltext_identity(text: str, seed: dict) -> dict:
    probe = text[:16_000].lower()
    compact_probe = _compact_identity_text(probe)
    doi = str(seed.get("doi") or "").lower().replace("https://doi.org/", "").strip()
    compact_doi = "".join(character for character in doi if not character.isspace())
    compact_probe_with_punctuation = "".join(character for character in probe if not character.isspace())
    if compact_doi and compact_doi in compact_probe_with_punctuation:
        return {"matched": True, "method": "doi", "score": 1.0}
    title = str(seed.get("title") or "").lower().strip()
    compact_title = _compact_identity_text(title)
    if len(compact_title) >= 8 and compact_title in compact_probe:
        return {"matched": True, "method": "title-exact", "score": 1.0}
    title_tokens = {
        token for token in re.findall(r"[a-z0-9]{4,}", title)
        if token not in {"with", "from", "this", "that", "using", "based", "paper", "study"}
    }
    if not title_tokens:
        return {"matched": False, "method": "title", "score": 0.0}
    matched = sum(1 for token in title_tokens if token in probe)
    score = matched / len(title_tokens)
    return {"matched": matched >= 3 and score >= 0.55, "method": "title", "score": round(score, 3)}


def _compact_identity_text(value: str) -> str:
    """Keep Unicode letters and numbers so line breaks and punctuation do not hide a title match."""
    return "".join(character for character in value.casefold() if character.isalnum())


def _merge_related_notes(report: dict, notes: dict) -> None:
    if not isinstance(notes, dict):
        return
    for paper in report.get("relatedPapers", []):
        note = notes.get(str(paper.get("id")))
        if not isinstance(note, dict):
            continue
        paper["summary"] = _clean_related_note_text(note.get("summary"), paper) or paper["summary"]
        paper["why"] = _clean_related_note_text(note.get("why"), paper) or paper["why"]
        paper["difference"] = _clean_related_note_text(note.get("difference"), paper) or paper.get("difference", "")
        paper["evidenceBoundary"] = _clean_related_note_text(note.get("evidenceBoundary"), paper) or paper.get("evidenceBoundary", "")
        paper["analysisLevel"] = "模型详解"


def _clean_related_note_text(value: object, paper: dict) -> str:
    text = str(value or "").strip()
    paper_id = str(paper.get("id") or "").strip()
    if text and paper_id:
        text = text.replace(paper_id, str(paper.get("shortTitle") or paper.get("title") or "该论文"))
        text = text.replace(paper_id.lower(), str(paper.get("shortTitle") or paper.get("title") or "该论文"))
        text = text.replace(paper_id.upper(), str(paper.get("shortTitle") or paper.get("title") or "该论文"))
    return text


def _deduplicate_papers(seed: dict, candidates: list[dict], limit: int) -> list[dict]:
    seen = set(_paper_keys(seed))
    result = []
    for paper in candidates:
        keys = _paper_keys(paper)
        if not keys or keys & seen or not paper.get("title"):
            continue
        seen.update(keys)
        result.append(paper)
        if len(result) >= limit:
            break
    return result


def _paper_key(paper: dict) -> str:
    return str(paper.get("openalexId") or paper.get("doi") or paper.get("title") or "").lower().strip()


def _paper_keys(paper: dict) -> set[str]:
    keys = set()
    openalex_id = str(paper.get("openalexId") or "").lower().strip()
    doi = str(paper.get("doi") or "").lower().replace("https://doi.org/", "").strip()
    title = "".join(character for character in str(paper.get("title") or "").lower() if character.isalnum())
    if openalex_id:
        keys.add(f"openalex:{openalex_id}")
    if doi:
        keys.add(f"doi:{doi}")
    if title:
        keys.add(f"title:{title}")
    return keys


def _display_paper(paper: dict, relation: str, why: str) -> dict:
    cluster = {
        "参考文献": "参考文献回溯",
        "后续引用": "后续引用扩展",
        "语义相关": "语义相关工作",
        "语义检索": "稀疏结果补充",
    }.get(relation, "种子论文")
    has_abstract = bool(paper.get("abstract"))
    abstract = paper.get("abstract") or "公开元数据未收录摘要，请打开原始来源进一步阅读。"
    return {
        "id": paper["id"],
        "title": paper["title"],
        "shortTitle": paper["shortTitle"],
        "authors": paper.get("authors", "作者信息未收录"),
        "year": paper.get("year") or 0,
        "venue": paper.get("venue") or "发表场所未收录",
        "relation": relation,
        "cluster": cluster,
        "summary": abstract[:900] + ("…" if len(abstract) > 900 else ""),
        "why": why,
        "difference": "",
        "evidenceBoundary": "",
        "abstractAvailable": has_abstract,
        "analysisLevel": "公开摘要" if has_abstract else "仅元数据",
        "confidence": "高" if relation in {"参考文献", "后续引用"} else "中",
        "sourceUrl": paper.get("sourceUrl") or "",
    }


def _is_echosight_demo_query(query: str) -> bool:
    probe = query.lower()
    return any(token in probe for token in ("echosight", "2024.findings-emnlp.83", "10.18653/v1/2024.findings-emnlp.83"))

