import os
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from backend.config import (
    LLM_PROVIDER_PRESETS,
    Settings,
    activate_runtime_llm,
    activate_runtime_openalex,
    deactivate_runtime_llm,
    deactivate_runtime_openalex,
    reset_runtime_llm,
    reset_runtime_openalex,
)
from backend.integrations import (
    DeepSeekLLMClient,
    IntegrationNotConfigured,
    OpenAICompatibleLLMClient,
    OpenAlexClient,
    ProviderBundle,
    ProviderRequestError,
    ZhipuLLMClient,
    build_provider_bundle,
    _aggregate_usage,
    _select_paper_material,
    _normalize_deep_dive_payload,
    _normalize_related_notes_payload,
    validate_report_patch,
    validate_related_notes_patch,
    validate_synthesis_patch,
)
from backend.evaluation import compare_scores, markdown_comparison, score_report
from backend.mock_data import build_demo_report
from backend.orchestrator import _build_live_report, _deduplicate_papers, _verify_fulltext_identity, fetch_open_access_fulltext, get_pipeline, run_research
from backend.pdf_text import download_open_access_pdf, extract_pdf_text
from backend.run_archive import RunArchive
from backend.server import _RESEARCH_JOBS, _RESEARCH_JOBS_LOCK, _llm_candidate, _openalex_candidate, _run_research_job
from backend.stage_cache import StageCache
from scripts.benchmark import _pair_materials, _prepare_material, _resolve_material_mode, _summarize_scores


class PaperScopeCoreTests(unittest.TestCase):
    def setUp(self):
        reset_runtime_llm()
        reset_runtime_openalex()

    def tearDown(self):
        reset_runtime_llm()
        reset_runtime_openalex()

    def test_demo_report_has_stable_schema(self):
        report = run_research("EchoSight", True, Settings.from_environment())
        self.assertEqual(report["meta"]["mode"], "demo")
        self.assertEqual(report["paper"]["id"], "echosight")
        self.assertGreaterEqual(len(report["relatedPapers"]), 8)
        self.assertGreaterEqual(len(report["relations"]), 7)
        self.assertEqual(len(report["pipeline"]), 6)

    def test_benchmark_scoring_is_deterministic_and_comparable(self):
        benchmark = {
            "id": "tiny",
            "title": "A Tiny Paper",
            "goldAnchors": [{"id": "method", "label": "方法锚点", "patterns": ["可靠性", "证据"]}],
        }
        report = build_demo_report("EchoSight")
        report["paper"].update({"title": "A Tiny Paper", "oneSentence": "这是一个足够具体且完整的一句话论文定位说明。"})
        report["analysis"]["problem"] = "这项工作研究可靠性与证据累积之间的关系，并讨论采样成本和停止规则。"
        report["analysis"]["deepDive"] = {
            "keyConcepts": [
                {"term": "概念一", "explanation": "这是第一个足够完整的关键概念说明。"},
                {"term": "概念二", "explanation": "这是第二个足够完整的关键概念说明。"},
                {"term": "概念三", "explanation": "这是第三个足够完整的关键概念说明。"},
            ],
            "technicalDetails": ["技术机制详细说明一", "技术机制详细说明二", "技术机制详细说明三"],
            "experimentalSetup": ["实验设置详细说明一", "实验设置详细说明二", "实验设置详细说明三"],
            "findings": ["主要实验发现详细说明一", "主要实验发现详细说明二", "主要实验发现详细说明三"],
            "readingGuide": ["首先阅读问题定义部分", "随后检查方法机制部分", "最后核对实验结论部分"],
        }
        report["meta"].update({"version": "test", "modelCall": {"verified": True}})
        for paper in report["relatedPapers"][1:6]:
            paper.update({
                "difference": "该工作与种子论文在证据累积方式上存在明确差异。",
                "evidenceBoundary": "该判断仅依据公开摘要与元数据，尚需原文复核。",
            })
        first = score_report(report, benchmark)
        second = score_report(report, benchmark)
        self.assertEqual(first["score"], second["score"])
        self.assertGreater(first["score"], 70)
        comparison = compare_scores(first, first)
        self.assertEqual(comparison["delta"], 0)

    def test_repeated_benchmark_summary_keeps_failures_and_medians(self):
        scores = [
            {"benchmarkId": "tiny", "score": 80, "runtime": {"usage": {"total_tokens": 100, "prompt_cache_hit_tokens": 20, "reasoning_tokens": 4}, "elapsedSeconds": 10, "modelCallCount": 3, "validationRetries": 0, "schemaFallbackCount": 0}},
            {"benchmarkId": "tiny", "score": 100, "runtime": {"usage": {"total_tokens": 200, "prompt_cache_hit_tokens": 60, "reasoning_tokens": 8}, "elapsedSeconds": 20, "modelCallCount": 4, "validationRetries": 1, "schemaFallbackCount": 0}},
            {"benchmarkId": "tiny", "status": "failed", "score": 0, "error": "测试失败", "runtime": {}},
        ]
        summary = _summarize_scores(scores, "candidate")
        self.assertEqual(summary["successfulRuns"], 2)
        self.assertEqual(summary["successRatePercent"], 66.67)
        self.assertEqual(summary["score"]["median"], 90)
        self.assertEqual(summary["totalTokens"]["median"], 150)
        self.assertEqual(summary["promptCacheHitTokens"]["median"], 40)
        self.assertEqual(summary["reasoningTokens"]["median"], 6)
        self.assertEqual(summary["failures"], ["测试失败"])

    def test_extended_usage_aggregates_deepseek_cache_and_reasoning_tokens(self):
        usage = _aggregate_usage([
            {"usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 5},
            }},
            {"usage": {
                "prompt_tokens": 40,
                "completion_tokens": 10,
                "total_tokens": 50,
                "prompt_cache_hit_tokens": 30,
                "prompt_cache_miss_tokens": 10,
                "reasoning_tokens": 2,
            }},
        ])
        self.assertEqual(usage["total_tokens"], 170)
        self.assertEqual(usage["prompt_cache_hit_tokens"], 110)
        self.assertEqual(usage["prompt_cache_miss_tokens"], 30)
        self.assertEqual(usage["reasoning_tokens"], 7)

    def test_numeric_grounding_rejects_unsupported_salient_values(self):
        payload = {
            "thesis": "材料报告该方法降低70%推理成本，并覆盖27B模型。",
            "gaps": [
                "公开摘要未说明公式细节。", "当前材料无法判断阈值设置。",
                "仍需原文核验消融实验。", "现有材料未说明失败案例。",
            ],
        }
        with self.assertRaises(ProviderRequestError):
            validate_synthesis_patch(payload, source_text="The method reduces cost by 40% on a 27B model.")
        validated = validate_synthesis_patch(payload, source_text="The method reduces cost by 70% on a 27B model.")
        self.assertIn("70%", validated["thesis"])

    def test_markdown_comparison_includes_cost_and_timing_deltas(self):
        comparison = {
            "benchmarkId": "reasc", "before": 80, "after": 90, "delta": 10,
            "categories": {},
            "runtime": {
                "before": {"materialScope": "公开摘要", "usage": {"total_tokens": 100, "prompt_cache_hit_tokens": 20}, "elapsedSeconds": 10},
                "after": {"materialScope": "开放全文", "usage": {"total_tokens": 180, "prompt_cache_hit_tokens": 90}, "elapsedSeconds": 16},
            },
        }
        rendered = markdown_comparison(comparison)
        self.assertIn("运行成本与材料", rendered)
        self.assertIn("| 总 Token | 100 | 180 | +80 |", rendered)
        self.assertIn("| Prompt Cache 命中 Token | 20 | 90 | +70 |", rendered)
        self.assertIn("优化后材料：开放全文", rendered)

    def test_short_deep_lists_receive_transparent_boundary_fallbacks(self):
        payload = {
            "deepDive": {
                "keyConcepts": [],
                "technicalDetails": ["模型只返回了一项技术机制说明"],
                "experimentalSetup": ["模型只返回了一项实验设置说明"],
                "findings": ["模型只返回了一项实验发现说明"],
                "readingGuide": ["模型只返回了一项阅读建议说明"],
            }
        }
        normalized, fallbacks = _normalize_deep_dive_payload(payload, required=True)
        self.assertEqual(len(normalized["deepDive"]["technicalDetails"]), 3)
        self.assertEqual(len(normalized["deepDive"]["experimentalSetup"]), 3)
        self.assertIn("experimentalSetup", fallbacks)
        self.assertEqual(payload["deepDive"]["experimentalSetup"], ["模型只返回了一项实验设置说明"])

    def test_standard_mode_repairs_partial_deep_lists_too(self):
        payload = {
            "deepDive": {
                "keyConcepts": [
                    {"term": "可靠性", "explanation": "用于衡量单条模型响应是否值得纳入证据累积。"},
                    {"term": "自适应停止", "explanation": "根据已累积证据判断是否还需要继续采样。"},
                    {"term": "证据加权", "explanation": "按响应可靠程度调整不同答案对最终决策的贡献。"},
                ],
                "technicalDetails": ["模型只返回了一项技术机制说明"],
                "experimentalSetup": ["模型只返回了一项实验设置说明"],
                "findings": ["模型只返回了一项实验发现说明"],
                "readingGuide": ["模型只返回了一项阅读建议说明"],
            }
        }
        normalized, fallbacks = _normalize_deep_dive_payload(payload, required=False)
        self.assertEqual(len(normalized["deepDive"]["experimentalSetup"]), 3)
        self.assertIn("experimentalSetup", fallbacks)

    def test_invalid_deep_list_shapes_are_recreated_or_trimmed(self):
        payload = {
            "deepDive": {
                "technicalDetails": None,
                "experimentalSetup": [f"实验设置详细说明第{index}项" for index in range(10)],
                "findings": ["发现一", "发现二", "发现三"],
                "readingGuide": ["指南一", "指南二", "指南三"],
            }
        }
        normalized, fallbacks = _normalize_deep_dive_payload(payload, required=True)
        self.assertEqual(len(normalized["deepDive"]["technicalDetails"]), 3)
        self.assertEqual(len(normalized["deepDive"]["experimentalSetup"]), 8)
        self.assertIn("technicalDetails:recreated", fallbacks)
        self.assertIn("experimentalSetup:trimmed", fallbacks)

    def test_related_work_deduplication_uses_title_across_openalex_records(self):
        seed = {"openalexId": "W1", "doi": "10.1/seed", "title": "Same Paper Title"}
        duplicate = {"openalexId": "W2", "doi": "10.1/version", "title": "Same Paper Title"}
        related = {"openalexId": "W3", "doi": "10.1/related", "title": "A Distinct Related Paper"}
        self.assertEqual(_deduplicate_papers(seed, [duplicate, related], limit=10), [related])

    def test_missing_related_notes_receive_visible_degraded_fallbacks(self):
        payload = {"relatedNotes": None}
        materials = [{"id": "w2", "title": "Related Work", "relation": "参考文献"}]
        normalized, fallbacks = _normalize_related_notes_payload(payload, ["w2"], materials)
        note = normalized["relatedNotes"][0]
        self.assertEqual(note["id"], "w2")
        self.assertIn("未能通过校验", note["summary"])
        self.assertIn("参考文献", note["evidenceBoundary"])
        self.assertEqual(len(fallbacks), 4)

    def test_blank_query_is_rejected(self):
        with self.assertRaises(ValueError):
            run_research("   ", True, Settings.from_environment())

    def test_oversized_source_material_is_rejected(self):
        with self.assertRaises(ValueError):
            run_research("EchoSight", True, Settings.from_environment(), source_text="材" * 80_001, depth="deep")

    def test_pdf_extraction_detects_research_sections_without_writing_files(self):
        fake_output = (
            b"Abstract\nA sufficiently long abstract about retrieval augmented visual question answering.\n\n"
            b"1 Introduction\nResearch motivation and problem definition.\n\n"
            b"3 Method\nDynamic retrieval and gated fusion details.\n\n"
            b"4 Experiments\nDatasets, metrics and results.\n\n"
            b"5 Conclusion\nSummary.\f"
        )

        class Result:
            returncode = 0
            stdout = fake_output
            stderr = b""

        with patch("backend.pdf_text.which", return_value="/usr/local/bin/pdftotext"), patch("backend.pdf_text.run", return_value=Result()) as mocked:
            extracted = extract_pdf_text(b"%PDF-1.7\nmock")
        self.assertEqual(extracted["pages"], 1)
        self.assertIn("方法", extracted["sections"])
        self.assertIn("实验", extracted["sections"])
        self.assertIn("Dynamic retrieval", extracted["text"])
        self.assertEqual(mocked.call_args.kwargs["input"], b"%PDF-1.7\nmock")

    def test_open_access_pdf_download_rejects_private_hosts_and_non_pdf(self):
        with self.assertRaises(ValueError):
            download_open_access_pdf("https://127.0.0.1/paper.pdf")

        class FakeResponse:
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def geturl(self):
                return "https://example.org/paper.pdf"

            def read(self, *args):
                return b"<html>login</html>"

        with patch("backend.pdf_text.urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(ValueError, "没有返回有效 PDF"):
                download_open_access_pdf("https://example.org/paper.pdf")

    def test_live_run_is_blocked_without_credentials(self):
        settings = Settings("", "", "", "", "", "", "", "")
        with self.assertRaises(IntegrationNotConfigured):
            run_research("10.1234/example", False, settings)

    def test_public_config_never_returns_secret(self):
        env = {
            "PAPERSCOPE_LLM_PROVIDER": "zhipu",
            "PAPERSCOPE_LLM_API_KEY": "super-secret",
            "PAPERSCOPE_LLM_BASE_URL": "https://example.test/v4",
            "PAPERSCOPE_LLM_MODEL": "model-x",
            "PAPERSCOPE_PAPER_SEARCH_PROVIDER": "search-x",
        }
        with patch.dict(os.environ, env, clear=True):
            public = Settings.from_environment().public_status()
        self.assertNotIn("super-secret", repr(public))
        self.assertTrue(public["llm"]["configured"])

    def test_pipeline_identifiers_are_unique(self):
        pipeline = get_pipeline()
        self.assertEqual(len({step["id"] for step in pipeline}), len(pipeline))

    def test_llm_only_configuration_uses_assisted_mode(self):
        settings = Settings("zhipu", "temporary", "https://example.test/v4", "glm-4.7-flash", "", "", "", "")
        self.assertEqual(settings.public_status()["mode"], "assisted")
        self.assertIsInstance(build_provider_bundle(settings).llm, ZhipuLLMClient)

    def test_free_paper_data_is_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_environment()
        public = settings.public_status()
        self.assertEqual(public["paperSearch"]["provider"], "openalex-free")
        self.assertEqual(public["citationGraph"]["mode"], "anonymous")
        bundle = build_provider_bundle(settings)
        self.assertIsInstance(bundle.paper_search, OpenAlexClient)
        self.assertIs(bundle.paper_search, bundle.citation_graph)

    def test_generic_openai_compatible_provider(self):
        settings = Settings("deepseek", "temporary", "https://api.example.test", "chat-model", "", "", "", "")
        client = build_provider_bundle(settings).llm
        self.assertIsInstance(client, OpenAICompatibleLLMClient)
        self.assertNotIsInstance(client, ZhipuLLMClient)

    def test_page_configuration_lives_only_in_process_memory(self):
        settings = Settings("zhipu", "temporary-secret", "https://example.test/v4", "glm-test", "", "", "", "")
        activate_runtime_llm(settings)
        active = Settings.from_environment()
        self.assertEqual(active.llm_model, "glm-test")
        self.assertNotIn("temporary-secret", repr(active.public_status()))
        self.assertEqual(active.public_status()["llm"]["storage"], "process-memory")
        deactivate_runtime_llm()
        self.assertFalse(Settings.from_environment().llm_ready)

    def test_page_configuration_validation(self):
        candidate = _llm_candidate({
            "provider": "zhipu",
            "apiKey": "temporary-key",
            "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4.7-flash",
        })
        self.assertTrue(candidate.llm_ready)
        with self.assertRaises(ValueError):
            _llm_candidate({"provider": "unknown", "apiKey": "temporary-key"})

    def test_openalex_key_lives_only_in_process_memory(self):
        activate_runtime_openalex("openalex-temporary-secret", {"remaining": 900, "limit": 1000})
        active = Settings.from_environment()
        public = active.public_status()
        self.assertEqual(active.paper_search_api_key, "openalex-temporary-secret")
        self.assertEqual(active.citation_api_key, "openalex-temporary-secret")
        self.assertEqual(public["paperSearch"]["mode"], "api-key")
        self.assertEqual(public["paperSearch"]["rateLimit"]["remaining"], 900)
        self.assertNotIn("openalex-temporary-secret", repr(public))
        deactivate_runtime_openalex()
        self.assertFalse(Settings.from_environment().paper_search_api_key)

    def test_openalex_page_key_validation(self):
        candidate = _openalex_candidate({"apiKey": "openalex-valid-key"})
        self.assertEqual(candidate.paper_search_api_key, "openalex-valid-key")
        self.assertEqual(candidate.citation_api_key, "openalex-valid-key")
        with self.assertRaises(ValueError):
            _openalex_candidate({"apiKey": "short"})

    def test_openalex_key_connection_reports_quota(self):
        class FakeResponse:
            headers = {
                "X-RateLimit-Limit": "1000",
                "X-RateLimit-Remaining": "875",
                "X-RateLimit-Reset": "3600",
            }

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "api_key": "masked",
                    "rate_limit": {
                        "credits_limit": 1000,
                        "credits_used": 125,
                        "credits_remaining": 875,
                        "resets_in_seconds": 3600,
                    },
                }).encode()

        client = OpenAlexClient(api_key="openalex-valid-key")
        with patch("backend.integrations.urlopen", return_value=FakeResponse()) as mocked:
            result = client.test_connection()
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "api-key")
        self.assertEqual(result["rateLimit"]["remaining"], 875)
        self.assertEqual(mocked.call_count, 1)

    def test_openalex_reuses_in_process_cache(self):
        class FakeResponse:
            headers = {"X-RateLimit-Remaining": "999"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"results": [{
                    "id": "https://openalex.org/W987654321",
                    "title": "Cached Paper",
                    "publication_year": 2025,
                    "authorships": [],
                    "primary_location": {"source": {}},
                    "referenced_works": [],
                    "related_works": [],
                    "topics": [],
                }]}).encode()

        client = OpenAlexClient()
        with patch("backend.integrations.urlopen", return_value=FakeResponse()) as mocked:
            first = client.resolve_paper("10.7654/paperscope-cache-test")
            second = client.resolve_paper("10.7654/paperscope-cache-test")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(client.rate_limit_status()["cacheHits"], 1)

    def test_report_patch_validation(self):
        payload = {
            "oneSentence": "这是符合长度要求的一句话总结。",
            "problem": "这是符合长度要求的研究问题描述。",
            "contributions": ["第一项有效贡献描述", "第二项有效贡献描述", "第三项有效贡献描述"],
            "methodSteps": [
                {"label": "检索", "text": "检索外部知识候选内容"},
                {"label": "筛选", "text": "筛选与问题相关的证据"},
                {"label": "回答", "text": "基于视觉和知识生成答案"},
            ],
            "evidence": ["第一项有效证据描述", "第二项有效证据描述", "第三项有效证据描述"],
            "limitations": ["第一项有效局限描述", "第二项有效局限描述", "第三项有效局限描述"],
            "thesis": "这是符合长度要求的综合判断。",
            "gaps": ["第一项研究空白描述", "第二项研究空白描述", "第三项研究空白描述", "第四项研究空白描述"],
            "deepDive": {
                "keyConcepts": [
                    {"term": "概念一", "explanation": "这是第一个关键概念的中文解释。"},
                    {"term": "概念二", "explanation": "这是第二个关键概念的中文解释。"},
                    {"term": "概念三", "explanation": "这是第三个关键概念的中文解释。"},
                ],
                "technicalDetails": ["第一项技术机制详细说明", "第二项技术机制详细说明", "第三项技术机制详细说明"],
                "experimentalSetup": ["第一项实验设置详细说明", "第二项实验设置详细说明", "第三项实验设置详细说明"],
                "findings": ["第一项主要实验发现说明", "第二项主要实验发现说明", "第三项主要实验发现说明"],
                "readingGuide": ["首先阅读问题定义部分", "然后核对方法模块关系", "最后检查实验与消融结果"],
            },
        }
        self.assertEqual(validate_report_patch(payload)["methodSteps"][0]["label"], "检索")
        self.assertEqual(len(validate_report_patch(payload, require_deep=True)["deepDive"]["keyConcepts"]), 3)
        payload["gaps"] = ["数量不足"]
        with self.assertRaises(ProviderRequestError):
            validate_report_patch(payload)

    def test_report_patch_rejects_english_and_placeholder_content(self):
        payload = {
            "oneSentence": "This is an English abstract copied directly from the source.",
            "problem": "这是符合要求的中文研究问题描述。",
            "contributions": ["第一项有效贡献描述", "第二项有效贡献描述", "第三项有效贡献描述"],
            "methodSteps": [
                {"label": "检索", "text": "使用多路径检索候选知识"},
                {"label": "融合", "text": "动态调整不同模态的权重"},
                {"label": "回答", "text": "依据筛选证据生成最终答案"},
            ],
            "evidence": ["摘要明确描述多路径检索", "摘要明确描述动态模态权重", "实验数字需要回到原文核验"],
            "limitations": ["免费数据未包含论文全文", "引用记录可能存在更新延迟", "语义关系不等同于真实引用"],
            "thesis": "结构化综合将在模型读取材料后生成。",
            "gaps": ["需要核验实验设置", "需要核验消融结果", "需要比较检索成本", "需要检查失败案例"],
        }
        with self.assertRaises(ProviderRequestError):
            validate_report_patch(payload)

    def test_related_notes_require_complete_chinese_coverage(self):
        payload = {
            "oneSentence": "该方法通过动态多路径检索整合互补知识。",
            "problem": "现有系统使用单一模态检索，难以覆盖互补且异构的外部知识。",
            "contributions": ["提出动态多路径检索框架", "设计按查询调整的模态权重", "统一处理异构证据来源"],
            "methodSteps": [
                {"label": "多路径检索", "text": "分别从文本和视觉路径检索候选知识"},
                {"label": "动态加权", "text": "依据查询调整各个模态路径的重要程度"},
                {"label": "证据融合", "text": "融合互补知识并用于视觉问题回答"},
            ],
            "evidence": ["摘要明确说明多路径检索机制", "摘要说明融合权重会随查询变化", "具体实验数字仍需查阅论文原文"],
            "limitations": ["公开摘要不能覆盖全部实现细节", "当前引用数据可能存在收录延迟", "语义检索结果仍需人工核验"],
            "thesis": "该工作把固定单路检索推进为面向查询的动态异构证据组织。",
            "gaps": ["需要比较不同路径的独立贡献", "需要分析动态权重稳定性", "需要统一评估检索成本", "需要检查失败样本"],
            "relatedNotes": [{
                "id": "w2",
                "summary": "该论文研究细粒度多模态知识检索，并通过跨模态交互组织候选证据以支持下游回答。",
                "why": "它为种子论文的外部知识检索模块提供直接的方法对照。",
                "difference": "该工作强调细粒度跨模态匹配，而种子论文关注动态组织多条检索路径。",
                "evidenceBoundary": "当前判断来自公开摘要与语义关系，具体实验差异仍需原文核验。",
            }],
        }
        validated = validate_report_patch(payload, expected_related_ids=["w2"])
        self.assertIn("w2", validated["relatedNotes"])
        payload["relatedNotes"][0].pop("difference")
        with self.assertRaises(ProviderRequestError):
            validate_report_patch(payload, expected_related_ids=["w2"])
        payload["relatedNotes"] = []
        with self.assertRaises(ProviderRequestError):
            validate_report_patch(payload, expected_related_ids=["w2"])

    def test_unbounded_negative_comparison_claims_are_rejected(self):
        related = {"relatedNotes": [{
            "id": "w2",
            "summary": "该工作研究自适应采样，并依据公开摘要整理其主要方法与结果边界。",
            "why": "它提供了自适应停止机制的直接方法参照。",
            "difference": "种子论文不依赖问题难度先验，而是采用响应级置信度。",
            "evidenceBoundary": "当前判断来自公开摘要，仍需原文核验。",
        }]}
        with self.assertRaises(ProviderRequestError):
            validate_related_notes_patch(related, ["w2"])
        related["relatedNotes"][0]["difference"] = "公开摘要未说明种子论文是否依赖问题难度先验，仍需原文核验。"
        self.assertIn("w2", validate_related_notes_patch(related, ["w2"]))

        synthesis = {
            "thesis": "种子论文不使用对数概率分布，而是采用响应级置信度完成聚合。",
            "gaps": [
                "公开摘要未说明公式细节。", "当前材料无法判断阈值设置。",
                "仍需原文核验消融实验。", "现有材料未说明失败案例。",
            ],
        }
        with self.assertRaises(ProviderRequestError):
            validate_synthesis_patch(synthesis, source_text="response confidence")

    def test_semantic_neighbors_cannot_be_presented_as_lineage(self):
        related = {"relatedNotes": [{
            "id": "w2",
            "summary": "该工作研究自适应采样，并依据公开摘要整理其主要方法与结果边界。",
            "why": "它为种子论文的响应级置信度设计提供直接的方法思想来源。",
            "difference": "两项工作使用不同信号形成采样停止条件，可作为机制对照。",
            "evidenceBoundary": "当前仅确认语义相关关系，双方是否存在引用仍需原文核验。",
        }]}
        with self.assertRaises(ProviderRequestError):
            validate_related_notes_patch(related, ["w2"], source_relations={"w2": "语义相关"})
        related["relatedNotes"][0]["why"] = "它是响应级置信度采样研究中的概念近邻，可作为机制对照。"
        self.assertIn(
            "w2",
            validate_related_notes_patch(related, ["w2"], source_relations={"w2": "语义相关"}),
        )

    def test_evidence_boundaries_count_as_report_limitations(self):
        report = {
            "analysis": {
                "limitations": [
                    "当前材料未说明完整实验设置，需要回到论文原文核验。",
                    "公开摘要无法判断不同模型上的阈值稳定性。",
                    "现有材料尚未覆盖失败案例与系统性过度自信场景。",
                ]
            }
        }
        score = score_report(report, {"id": "tiny", "title": "", "goldAnchors": []})
        self.assertTrue(score["categories"]["structure"]["checks"]["至少三项局限"])

    def test_pair_order_can_be_counterbalanced(self):
        self.assertEqual(_pair_materials("abstract-first")[0], ("abstract", "abstract"))
        self.assertEqual(_pair_materials("fulltext-first")[0], ("fulltext", "auto-oa"))

    def test_run_archive_removes_raw_material_and_secrets(self):
        report = {
            "meta": {"mode": "live", "provider": "deepseek", "model": "deepseek-flash", "usage": {"total_tokens": 123}},
            "paper": {"title": "A Paper"},
            "_sourceMaterial": {"fullText": "raw paper text"},
            "debug": {"apiKey": "secret", "sourceText": "raw request text", "safe": "kept"},
        }
        with TemporaryDirectory() as directory:
            archive = RunArchive(Path(directory), max_entries=4)
            record = archive.save_completed(report, job_id="job-1", elapsed_seconds=12.3456)
            payload = json.loads((Path(directory) / f"{record['archiveId']}.json").read_text(encoding="utf-8"))
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("raw paper text", serialized)
            self.assertNotIn("raw request text", serialized)
            self.assertNotIn("secret", serialized)
            self.assertEqual(payload["report"]["debug"]["safe"], "kept")
            self.assertEqual(archive.list_recent()[0]["totalTokens"], 123)

    def test_benchmark_material_modes_build_explicit_source_boundaries(self):
        benchmark = {"query": "10.1/example", "depth": "deep"}
        self.assertEqual(_resolve_material_mode(None, None), "abstract")
        self.assertEqual(_resolve_material_mode(None, "paper.pdf"), "source")
        text, source = _prepare_material("http://127.0.0.1:8765", benchmark, "abstract", None)
        self.assertEqual(text, "")
        self.assertEqual(source["kind"], "abstract-fallback")
        with patch("scripts.benchmark._request_json", return_value={
            "text": "FULL PAPER",
            "source": {"kind": "open-access-pdf", "url": "https://example.org/paper.pdf"},
        }) as mocked:
            text, source = _prepare_material("http://127.0.0.1:8765", benchmark, "auto-oa", None)
        self.assertEqual(text, "FULL PAPER")
        self.assertEqual(source["kind"], "open-access-pdf")
        self.assertTrue(mocked.call_args.args[0].endswith("/api/fulltext/open-access"))

    def test_live_validation_rejects_unsupported_gap_claims(self):
        payload = {
            "oneSentence": "该方法使用动态检索组织外部知识。",
            "problem": "现有系统难以根据问题选择互补的外部知识。",
            "contributions": ["提出动态检索框架", "设计问题感知加权机制", "统一组织异构知识"],
            "methodSteps": [
                {"label": "多路检索", "text": "通过多条路径检索异构候选知识"},
                {"label": "动态加权", "text": "根据问题调整各模态的重要程度"},
                {"label": "证据融合", "text": "融合筛选后的知识生成最终答案"},
            ],
            "evidence": ["摘要明确描述多路径检索", "摘要说明权重随问题变化", "实验数字仍需查阅原文"],
            "limitations": ["公开摘要不含全部实现细节", "引用记录可能存在收录延迟", "语义关系仍需人工核验"],
            "thesis": "该工作将固定检索推进为问题感知的异构知识选择。",
            "gaps": ["缺乏消融实验", "缺乏效率分析", "缺乏失败案例", "缺乏泛化实验"],
            "relatedNotes": [],
        }
        with self.assertRaises(ProviderRequestError):
            validate_report_patch(payload, expected_related_ids=[], source_text="The method uses dynamic retrieval.")

    def test_llm_retries_once_after_language_validation_failure(self):
        invalid = {
            "oneSentence": "Copied English abstract without a Chinese summary.",
            "problem": "Copied English problem statement without translation.",
            "contributions": ["English contribution one", "English contribution two", "English contribution three"],
            "methodSteps": [{"label": "A", "text": "English method step"}] * 3,
            "evidence": ["English evidence one", "English evidence two", "English evidence three"],
            "limitations": ["English limitation one", "English limitation two", "English limitation three"],
            "thesis": "Placeholder synthesis in English.",
            "gaps": ["English gap one", "English gap two", "English gap three", "English gap four"],
            "relatedNotes": [],
        }
        valid = {
            "oneSentence": "该方法通过动态检索组织外部知识。",
            "problem": "现有方法难以根据问题动态选择互补的外部知识。",
            "contributions": ["提出动态多路径检索框架", "设计查询感知的模态加权机制", "统一组织异构外部知识"],
            "methodSteps": [
                {"label": "检索", "text": "通过多条路径检索异构候选知识"},
                {"label": "加权", "text": "依据问题动态调整不同模态的重要性"},
                {"label": "回答", "text": "融合筛选后的证据生成最终答案"},
            ],
            "evidence": ["摘要明确描述多路径检索", "摘要说明权重会随问题变化", "具体实验数字仍需原文核验"],
            "limitations": ["摘要未覆盖全部实现细节", "引用数据可能存在收录延迟", "语义相关关系仍需人工核验"],
            "deepDive": {
                "keyConcepts": [
                    {"term": "动态检索", "explanation": "根据问题内容选择互补的外部知识来源。"},
                    {"term": "模态加权", "explanation": "按查询调整不同模态证据的重要程度。"},
                    {"term": "证据融合", "explanation": "将筛选后的异构知识组织为回答依据。"},
                ],
                "technicalDetails": ["执行多路径候选检索", "按问题调整模态权重", "融合证据生成最终答案"],
            },
            "thesis": "该工作将固定检索推进为查询感知的异构证据选择。",
            "gaps": ["公开摘要未说明各检索路径贡献", "当前材料无法判断动态权重稳定性", "仍需原文核验额外检索成本", "公开摘要未说明失败案例"],
            "relatedNotes": [],
        }
        responses = [
            {"content": json.dumps(invalid), "model": "glm-test", "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, "responseId": "rsp-1", "requestId": "req-1", "created": 100},
            {"content": json.dumps(valid, ensure_ascii=False), "model": "glm-test", "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}, "responseId": "rsp-2", "requestId": "req-2", "created": 101},
            {"content": json.dumps({"thesis": valid["thesis"], "gaps": valid["gaps"]}, ensure_ascii=False), "model": "glm-test", "usage": {"prompt_tokens": 6, "completion_tokens": 7, "total_tokens": 13}, "responseId": "rsp-3", "requestId": "req-3", "created": 102},
        ]
        client = OpenAICompatibleLLMClient("temporary", "glm-test", "https://example.test", "openai", use_stage_cache=False)
        report = {
            "paper": {"id": "seed", "title": "Seed", "authors": "A", "year": 2026, "venue": "V", "identifier": "I"},
            "analysis": {"problem": "问题", "contributions": []},
            "relatedPapers": [{"id": "seed", "title": "Seed", "year": 2026, "venue": "V", "relation": "种子论文", "summary": "摘要"}],
            "_sourceMaterial": {"abstract": "Source abstract."},
        }
        with patch.object(client, "_complete", side_effect=responses) as mocked:
            result = client.enhance_report(report)
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(result["_provider"]["validationRetries"], 1)
        self.assertEqual(result["_provider"]["usage"]["total_tokens"], 25)
        self.assertEqual(result["_provider"]["responseId"], "rsp-3")
        self.assertEqual(len(result["_provider"]["calls"]), 3)
        self.assertEqual([item["stage"] for item in result["_provider"]["calls"]], ["seed-core", "seed-core", "synthesis"])
        self.assertIn("动态检索", result["oneSentence"])

    def test_llm_runs_seed_related_and_synthesis_as_separate_stages(self):
        seed_payload = {
            "oneSentence": "该方法通过可靠性加权证据实现自适应采样决策。",
            "problem": "固定采样预算会对简单问题产生不必要成本，也无法根据不同响应的可靠程度调整停止时机。",
            "contributions": ["提出可靠性感知采样框架", "设计置信度加权证据累积", "验证准确率与成本权衡"],
            "methodSteps": [
                {"label": "置信度估计", "text": "计算每条推理响应的可靠性信号"},
                {"label": "证据累积", "text": "按照置信度加权累积候选答案证据"},
                {"label": "自适应停止", "text": "证据充分后停止继续生成推理样本"},
            ],
            "evidence": ["摘要明确说明两阶段框架", "摘要报告准确率成本权衡", "具体设置仍需查阅原文"],
            "limitations": ["公开摘要未覆盖全部公式", "置信度校准仍需原文核验", "不同任务表现仍需进一步比较"],
            "deepDive": {
                "keyConcepts": [
                    {"term": "响应置信度", "explanation": "衡量单条响应是否值得纳入最终证据累积。"},
                    {"term": "证据累积", "explanation": "联合响应频率与可靠程度更新候选答案支持度。"},
                    {"term": "自适应停止", "explanation": "累积证据充分时停止继续生成推理样本。"},
                ],
                "technicalDetails": ["估计响应可靠性信号", "按置信度加权累积证据", "证据充分后提前停止采样"],
            },
        }
        related_payload = {"relatedNotes": [{
            "id": "w2",
            "summary": "该工作研究以回答频率为核心信号的自适应采样，并在证据达到阈值时决定是否停止继续生成。",
            "why": "它是种子论文比较的计数式自适应采样基线。",
            "difference": "公开摘要未说明该工作使用响应级可靠性加权证据，仍需原文核验。",
            "evidenceBoundary": "差异判断依据当前公开摘要，仍需原文复核。",
        }]}
        synthesis_payload = {
            "thesis": "种子论文将计数式停止推进为可靠性感知的证据累积。",
            "gaps": ["公开摘要未说明全部校准细节", "当前材料无法判断跨任务稳定性", "仍需原文核验计算开销", "公开摘要未说明失败案例"],
        }
        responses = [
            {"content": json.dumps(seed_payload, ensure_ascii=False), "model": "glm-test", "usage": {"total_tokens": 10}, "responseId": "seed"},
            {"content": json.dumps(related_payload, ensure_ascii=False), "model": "glm-test", "usage": {"total_tokens": 11}, "responseId": "related"},
            {"content": json.dumps(synthesis_payload, ensure_ascii=False), "model": "glm-test", "usage": {"total_tokens": 12}, "responseId": "synthesis"},
        ]
        report = {
            "paper": {"id": "w1", "title": "Seed", "authors": "A", "year": 2026, "venue": "V", "identifier": "I"},
            "analysis": {"problem": "问题", "contributions": []},
            "relatedPapers": [
                {"id": "w1", "title": "Seed", "year": 2026, "venue": "V", "relation": "种子论文", "summary": "Seed abstract"},
                {"id": "w2", "title": "Related", "year": 2024, "venue": "V", "relation": "参考文献", "summary": "该方法根据候选答案出现频率决定是否继续采样。"},
            ],
            "_sourceMaterial": {"abstract": "The method uses reliability-aware adaptive sampling.", "analysisDepth": "standard"},
        }
        client = OpenAICompatibleLLMClient("temporary", "glm-test", "https://example.test", "openai", use_stage_cache=False)
        with patch.object(client, "_complete", side_effect=responses) as mocked:
            result = client.enhance_report(report)
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(result["_provider"]["stageCount"], 3)
        self.assertEqual([item["stage"] for item in result["_provider"]["calls"]], ["seed-core", "related-work", "synthesis"])
        self.assertIn("w2", result["relatedNotes"])
        self.assertEqual(result["_provider"]["usage"]["total_tokens"], 33)

    def test_section_aware_material_prefers_relevant_sections_and_excludes_references(self):
        source = """Abstract
This paper studies adaptive sampling.

1 Introduction
The introduction explains the problem.

2 Method
The method uses confidence weighted evidence.

3 Experiments
We evaluate five models on four datasets.

4 Results
The method reduces inference cost while preserving accuracy.

References
[1] This reference text must not enter model material.
"""
        core = _select_paper_material(source, "core", 10_000)
        evidence = _select_paper_material(source, "evidence", 10_000)
        self.assertIn("confidence weighted evidence", core)
        self.assertIn("five models on four datasets", evidence)
        self.assertNotIn("must not enter model material", core)
        self.assertNotIn("must not enter model material", evidence)

    def test_deep_full_text_run_adds_independent_seed_evidence_stage(self):
        seed_payload = {
            "oneSentence": "该方法利用响应置信度判断证据充分性并自适应停止采样。",
            "problem": "固定采样预算忽略问题难度与响应可靠程度，容易造成不必要的推理开销。",
            "contributions": ["提出可靠性感知采样框架", "设计置信度加权证据累积", "验证准确率与成本权衡"],
            "methodSteps": [
                {"label": "置信估计", "text": "计算每条推理响应的可靠性信号"},
                {"label": "证据累积", "text": "联合频率与置信度累积答案证据"},
                {"label": "自适应停止", "text": "证据充分后停止生成新的推理样本"},
            ],
            "evidence": ["摘要说明两阶段框架", "摘要报告成本与准确率权衡", "完整数字需由实验章节核验"],
            "limitations": ["摘要未提供全部公式", "阈值设置需要原文核验", "跨任务边界仍需进一步检查"],
            "deepDive": {
                "keyConcepts": [
                    {"term": "响应置信度", "explanation": "衡量单条模型响应可否作为可靠证据。"},
                    {"term": "证据充分性", "explanation": "判断当前累积信息是否足以支持停止采样。"},
                    {"term": "自适应停止", "explanation": "根据证据状态动态决定是否继续采样。"},
                ],
                "technicalDetails": ["模型先估计每条响应的置信度信号", "联合答案频率和响应置信度累积证据", "达到证据充分条件后停止继续采样"],
            },
        }
        evidence_payload = {
            "evidence": ["论文在五个模型上进行实验", "论文覆盖四个推理数据集", "实验报告推理成本显著下降"],
            "limitations": ["部分阈值需要人工设定", "当前实验未覆盖所有模型系列", "失败案例仍需查看论文附录"],
            "experimentalSetup": ["使用五个不同规模语言模型", "在四个推理数据集上评测", "同时比较准确率和推理成本"],
            "findings": ["方法保持主要任务准确率", "方法降低平均推理采样成本", "消融支持置信度信号具有贡献"],
            "readingGuide": ["先核对方法章节的停止公式", "再检查主结果表中的成本定义", "最后阅读消融与失败案例"],
        }
        synthesis_payload = {
            "thesis": "该工作将计数式停止推进为可靠性感知的证据累积。",
            "gaps": ["当前材料未说明全部阈值细节", "仍需原文核验跨模型稳定性", "当前材料无法判断部署开销", "当前材料未完整说明失败案例"],
        }
        responses = [
            {"content": json.dumps(seed_payload, ensure_ascii=False), "model": "test", "usage": {"total_tokens": 10}},
            {"content": json.dumps(evidence_payload, ensure_ascii=False), "model": "test", "usage": {"total_tokens": 11}},
            {"content": json.dumps(synthesis_payload, ensure_ascii=False), "model": "test", "usage": {"total_tokens": 12}},
        ]
        report = {
            "paper": {"id": "seed", "title": "Seed", "authors": "A", "year": 2026, "venue": "V", "identifier": "DOI"},
            "analysis": {"problem": "问题", "contributions": []},
            "relatedPapers": [{"id": "seed", "title": "Seed", "year": 2026, "venue": "V", "relation": "种子论文", "summary": "摘要"}],
            "_sourceMaterial": {
                "abstract": "The method uses confidence-aware adaptive sampling.",
                "analysisDepth": "deep",
                "fullText": "1 Introduction\nProblem context.\n2 Method\nConfidence weighted evidence.\n3 Experiments\nFive models and four datasets.\n4 Results\nLower inference cost.",
            },
        }
        client = OpenAICompatibleLLMClient("temporary", "test", "https://example.test", "openai", use_stage_cache=False)
        with patch.object(client, "_complete", side_effect=responses) as mocked:
            result = client.enhance_report(report)
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual([item["stage"] for item in result["_provider"]["calls"]], ["seed-core", "seed-evidence", "synthesis"])
        self.assertEqual(result["_provider"]["stageCount"], 3)
        self.assertEqual(result["deepDive"]["experimentalSetup"], evidence_payload["experimentalSetup"])
        self.assertEqual(result["evidence"], evidence_payload["evidence"])

    def test_stage_cache_resumes_after_third_stage_failure(self):
        seed_payload = {
            "oneSentence": "该方法通过可靠性加权证据实现自适应采样决策。",
            "problem": "固定采样预算会产生不必要成本，也无法依据不同响应的可靠程度调整停止时机。",
            "contributions": ["提出可靠性感知采样框架", "设计置信度加权证据累积", "验证准确率与成本权衡"],
            "methodSteps": [
                {"label": "置信度估计", "text": "计算每条推理响应的可靠性信号"},
                {"label": "证据累积", "text": "按照置信度加权累积候选答案证据"},
                {"label": "自适应停止", "text": "证据充分后停止继续生成推理样本"},
            ],
            "evidence": ["摘要明确说明两阶段框架", "摘要报告准确率成本权衡", "具体设置仍需查阅原文"],
            "limitations": ["公开摘要未覆盖全部公式", "置信度校准仍需原文核验", "不同任务表现仍需进一步比较"],
            "deepDive": {
                "keyConcepts": [
                    {"term": "响应置信度", "explanation": "衡量单条响应是否值得纳入最终证据累积。"},
                    {"term": "证据累积", "explanation": "联合响应频率与可靠程度更新候选答案支持度。"},
                    {"term": "自适应停止", "explanation": "累积证据充分时停止继续生成推理样本。"},
                ],
                "technicalDetails": ["估计响应可靠性信号", "按置信度加权累积证据", "证据充分后提前停止采样"],
            },
        }
        related_payload = {"relatedNotes": [{
            "id": "w2",
            "summary": "该工作研究以回答频率为核心信号的自适应采样，并在证据达到阈值时决定是否停止继续生成。",
            "why": "它是种子论文比较的计数式自适应采样基线。",
            "difference": "公开摘要未说明该工作使用响应级可靠性加权证据，仍需原文核验。",
            "evidenceBoundary": "差异判断依据当前公开摘要，仍需原文复核。",
        }]}
        synthesis_payload = {
            "thesis": "种子论文将计数式停止推进为可靠性感知的证据累积。",
            "gaps": ["公开摘要未说明全部校准细节", "当前材料无法判断跨任务稳定性", "仍需原文核验计算开销", "公开摘要未说明失败案例"],
        }
        report = {
            "paper": {"id": "w1", "title": "Seed", "authors": "A", "year": 2026, "venue": "V", "identifier": "I"},
            "analysis": {"problem": "问题", "contributions": []},
            "relatedPapers": [
                {"id": "w1", "title": "Seed", "year": 2026, "venue": "V", "relation": "种子论文", "summary": "Seed abstract"},
                {"id": "w2", "title": "Related", "year": 2024, "venue": "V", "relation": "参考文献", "summary": "An adaptive sampling baseline based on answer frequency."},
            ],
            "_sourceMaterial": {"abstract": "The method uses reliability-aware adaptive sampling.", "analysisDepth": "standard"},
        }
        seed_response = {"content": json.dumps(seed_payload, ensure_ascii=False), "model": "glm-test", "usage": {"total_tokens": 10}, "responseId": "seed"}
        related_response = {"content": json.dumps(related_payload, ensure_ascii=False), "model": "glm-test", "usage": {"total_tokens": 11}, "responseId": "related"}
        synthesis_response = {"content": json.dumps(synthesis_payload, ensure_ascii=False), "model": "glm-test", "usage": {"total_tokens": 12}, "responseId": "synthesis"}
        with TemporaryDirectory() as directory:
            cache = StageCache(Path(directory))
            first = OpenAICompatibleLLMClient("secret-never-cache", "glm-test", "https://example.test", "openai", stage_cache=cache)
            with patch.object(first, "_complete", side_effect=[seed_response, related_response, ProviderRequestError("第三阶段暂时失败")]) as mocked:
                with self.assertRaises(ProviderRequestError):
                    first.enhance_report(report)
            self.assertEqual(mocked.call_count, 3)
            self.assertEqual(cache.status()["entries"], 2)
            self.assertNotIn("secret-never-cache", "".join(path.read_text() for path in Path(directory).glob("*.json")))

            resumed = OpenAICompatibleLLMClient("another-secret", "glm-test", "https://example.test", "openai", stage_cache=cache)
            with patch.object(resumed, "_complete", return_value=synthesis_response) as mocked:
                result = resumed.enhance_report(report)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(result["_provider"]["cacheHits"], 2)
            self.assertEqual(result["_provider"]["resumedStages"], ["seed-core", "related-work"])
            self.assertEqual(result["_provider"]["usage"]["total_tokens"], 12)
            self.assertEqual(cache.status()["entries"], 3)

            cached = OpenAICompatibleLLMClient("third-secret", "glm-test", "https://example.test", "openai", stage_cache=cache)
            with patch.object(cached, "_complete", side_effect=AssertionError("不应再次调用模型")) as mocked:
                cached_result = cached.enhance_report(report)
            self.assertEqual(mocked.call_count, 0)
            self.assertEqual(cached_result["_provider"]["cacheHits"], 3)
            self.assertEqual(cached_result["_provider"]["usage"], {})
            cleared = cache.clear()
            self.assertEqual(cleared["removed"], 3)
            self.assertEqual(cleared["entries"], 0)

    def test_stage_cache_key_changes_with_model_material_and_prompt(self):
        with TemporaryDirectory() as directory:
            cache = StageCache(Path(directory))
            common = {
                "provider": "openai",
                "base_url": "https://example.test",
                "stage": "seed-analysis",
                "system_prompt": "system",
                "max_tokens": 100,
            }
            first = cache.make_key(model="model-a", user_prompt="paper-a", **common)
            self.assertNotEqual(first, cache.make_key(model="model-b", user_prompt="paper-a", **common))
            self.assertNotEqual(first, cache.make_key(model="model-a", user_prompt="paper-b", **common))
            self.assertNotEqual(first, cache.make_key(model="model-a", user_prompt="paper-a changed", **common))

    def test_assisted_run_merges_model_output_without_searching(self):
        patch_payload = {
            "oneSentence": "模型生成的一句话总结，长度足够。",
            "problem": "模型生成的研究问题描述，长度足够。",
            "contributions": ["模型生成贡献一描述", "模型生成贡献二描述", "模型生成贡献三描述"],
            "methodSteps": [
                {"label": "步骤一", "text": "模型生成步骤一的详细描述"},
                {"label": "步骤二", "text": "模型生成步骤二的详细描述"},
                {"label": "步骤三", "text": "模型生成步骤三的详细描述"},
            ],
            "evidence": ["模型生成证据一描述", "模型生成证据二描述", "模型生成证据三描述"],
            "limitations": ["模型生成局限一描述", "模型生成局限二描述", "模型生成局限三描述"],
            "thesis": "模型生成的综合判断内容，长度足够。",
            "gaps": ["模型生成空白一描述", "模型生成空白二描述", "模型生成空白三描述", "模型生成空白四描述"],
            "_provider": {"name": "zhipu", "label": "智谱 AI", "model": "glm-4.7-flash", "usage": {"total_tokens": 42}},
        }

        class FakeLlm:
            def enhance_report(self, report):
                return dict(patch_payload)

        settings = Settings("zhipu", "temporary", "https://example.test/v4", "glm-4.7-flash", "", "", "", "")
        with patch("backend.orchestrator.build_provider_bundle", return_value=ProviderBundle(llm=FakeLlm())):
            report = run_research("EchoSight", False, settings)
        self.assertEqual(report["meta"]["mode"], "llm-assisted-demo")
        self.assertEqual(report["paper"]["oneSentence"], patch_payload["oneSentence"])
        self.assertEqual(report["meta"]["usage"]["total_tokens"], 42)

    def test_assisted_run_rejects_unknown_paper_without_search_provider(self):
        settings = Settings("zhipu", "temporary", "https://example.test/v4", "glm-4.7-flash", "", "", "", "")
        with self.assertRaises(IntegrationNotConfigured):
            run_research("A completely different paper", False, settings)

    def test_zhipu_connection_check_fails_fast_without_retrying(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(self.payload).encode()

        responses = [FakeResponse({"error": {"code": 500, "message": "temporary"}})]
        client = ZhipuLLMClient("temporary", "glm-4.7-flash", "https://example.test/v4")
        with patch("backend.integrations.urlopen", side_effect=responses) as mocked, patch("backend.integrations.sleep"):
            with self.assertRaises(ProviderRequestError):
                client.test_connection()
        self.assertEqual(mocked.call_count, 1)
        request_body = json.loads(mocked.call_args_list[0].args[0].data)
        self.assertEqual(request_body["thinking"], {"type": "disabled"})
        self.assertEqual(request_body["max_tokens"], 16)

    def test_deepseek_preset_disables_thinking_and_uses_current_model(self):
        class FakeResponse:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "id": "deepseek-response",
                    "model": "deepseek-flash",
                    "choices": [{"message": {"content": "PAPER_SCOPE_OK"}}],
                    "usage": {"total_tokens": 12},
                }).encode()

        preset = next(item for item in LLM_PROVIDER_PRESETS if item["id"] == "deepseek")
        self.assertEqual(LLM_PROVIDER_PRESETS[0]["id"], "deepseek")
        self.assertEqual(preset["defaultModel"], "deepseek-flash")
        client = DeepSeekLLMClient("temporary", preset["defaultModel"], preset["baseUrl"])
        with patch("backend.integrations.urlopen", return_value=FakeResponse()) as mocked:
            result = client.test_connection()
        self.assertTrue(result["ok"])
        request_body = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(request_body["model"], "deepseek-flash")
        self.assertEqual(request_body["thinking"], {"type": "disabled"})
        self.assertEqual(request_body["max_tokens"], 64)

    def test_deepseek_insufficient_balance_is_quota_error_without_retry(self):
        client = DeepSeekLLMClient("temporary", "deepseek-flash", "https://api.deepseek.com")
        error = self._http_error(402, "Insufficient Balance")
        with patch("backend.integrations.urlopen", side_effect=error) as mocked:
            with self.assertRaises(ProviderRequestError) as raised:
                client.test_connection()
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(raised.exception.category, "quota")
        self.assertFalse(raised.exception.retryable)

    @staticmethod
    def _http_error(status, message, retry_after="0"):
        headers = {"Retry-After": retry_after} if retry_after is not None else {}
        body = json.dumps({"error": {"message": message}}).encode()
        return HTTPError("https://example.test", status, message, headers, BytesIO(body))

    def test_llm_quota_error_does_not_retry(self):
        client = OpenAICompatibleLLMClient("temporary", "model", "https://example.test", "openai", use_stage_cache=False)
        error = self._http_error(429, "账户余额不足，额度已用尽")
        with patch("backend.integrations.urlopen", side_effect=error) as mocked, patch("backend.integrations.sleep"):
            with self.assertRaises(ProviderRequestError) as raised:
                client._complete([{"role": "user", "content": "test"}], 16, False)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(raised.exception.category, "quota")
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(raised.exception.attempts, 1)

    def test_llm_congestion_retries_only_once(self):
        client = OpenAICompatibleLLMClient("temporary", "model", "https://example.test", "openai", use_stage_cache=False)
        errors = [
            self._http_error(429, "该模型当前访问量过大，请稍后再试"),
            self._http_error(429, "该模型当前访问量过大，请稍后再试"),
        ]
        with patch("backend.integrations.urlopen", side_effect=errors) as mocked, patch("backend.integrations.sleep") as sleeper:
            with self.assertRaises(ProviderRequestError) as raised:
                client._complete([{"role": "user", "content": "test"}], 16, False)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(sleeper.call_count, 1)
        self.assertEqual(raised.exception.category, "congestion")
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(raised.exception.attempts, 2)

    def test_llm_service_error_has_bounded_retry_and_attempt_count(self):
        class FakeResponse:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "model": "model",
                    "choices": [{"message": {"content": "PAPER_SCOPE_OK"}}],
                    "usage": {"total_tokens": 3},
                }).encode()

        client = OpenAICompatibleLLMClient("temporary", "model", "https://example.test", "openai", use_stage_cache=False)
        responses = [self._http_error(503, "temporary"), self._http_error(503, "temporary"), FakeResponse()]
        with patch("backend.integrations.urlopen", side_effect=responses) as mocked, patch("backend.integrations.sleep"):
            result = client._complete([{"role": "user", "content": "test"}], 16, False)
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(result["transportAttempts"], 3)

    def test_failed_job_keeps_provider_category_and_resume_state(self):
        job_id = "test-failed-job"
        with _RESEARCH_JOBS_LOCK:
            _RESEARCH_JOBS[job_id] = {
                "jobId": job_id, "status": "queued", "stepId": "resolve", "detail": "queued",
                "sequence": 0, "createdAt": 0, "updatedAt": 0,
            }
        error = ProviderRequestError(
            "模型拥堵",
            category="congestion",
            status=429,
            retryable=True,
            retry_after=2,
            attempts=2,
        )
        try:
            with patch("backend.server.run_research", side_effect=error):
                _run_research_job(job_id, {"query": "Paper", "demo": False}, Settings.from_environment())
            with _RESEARCH_JOBS_LOCK:
                job = dict(_RESEARCH_JOBS[job_id])
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["stepId"], "resolve")
            self.assertEqual(job["detail"], "模型服务拥堵")
            self.assertTrue(job["canResume"])
            self.assertEqual(job["failure"]["retryAfter"], 2)
            self.assertEqual(job["failure"]["attempts"], 2)
        finally:
            with _RESEARCH_JOBS_LOCK:
                _RESEARCH_JOBS.pop(job_id, None)

    def test_demo_progress_comes_from_orchestrator_events(self):
        events = []
        report = run_research(
            "EchoSight",
            True,
            Settings.from_environment(),
            progress=lambda step_id, detail: events.append((step_id, detail)),
        )
        self.assertEqual([item[0] for item in events], ["resolve", "synthesize"])
        self.assertEqual(report["pipeline"], get_pipeline())

    def test_background_research_job_finishes_with_report(self):
        job_id = "test-background-job"
        with _RESEARCH_JOBS_LOCK:
            _RESEARCH_JOBS[job_id] = {
                "jobId": job_id,
                "status": "queued",
                "stepId": "resolve",
                "detail": "queued",
                "sequence": 0,
                "createdAt": 0,
                "updatedAt": 0,
            }
        try:
            _run_research_job(
                job_id,
                {"query": "EchoSight", "demo": True, "depth": "standard", "sourceText": ""},
                Settings.from_environment(),
            )
            with _RESEARCH_JOBS_LOCK:
                job = dict(_RESEARCH_JOBS[job_id])
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["stepId"], "synthesize")
            self.assertTrue(job["report"]["paper"]["title"].startswith("EchoSight:"))
            self.assertGreater(job["sequence"], 0)
        finally:
            with _RESEARCH_JOBS_LOCK:
                _RESEARCH_JOBS.pop(job_id, None)

    def test_generic_provider_uses_portable_request_body(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "id": "response-123",
                    "request_id": "request-456",
                    "created": 1234567890,
                    "model": "chat-model",
                    "choices": [{"message": {"content": "PAPER_SCOPE_OK"}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 8},
                }).encode()

        client = OpenAICompatibleLLMClient("temporary", "chat-model", "https://example.test/v1", "openai")
        with patch("backend.integrations.urlopen", return_value=FakeResponse()) as mocked:
            connection = client.test_connection()
            self.assertTrue(connection["ok"])
        self.assertEqual(connection["responseId"], "response-123")
        self.assertEqual(connection["requestId"], "request-456")
        request_body = json.loads(mocked.call_args.args[0].data)
        self.assertNotIn("thinking", request_body)
        self.assertNotIn("do_sample", request_body)

    def test_openalex_doi_resolution_normalizes_metadata(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"results": [{
                    "id": "https://openalex.org/W123",
                    "doi": "https://doi.org/10.1234/test",
                    "title": "A Test Paper",
                    "publication_year": 2025,
                    "authorships": [{"author": {"display_name": "Ada Example"}}],
                    "primary_location": {"landing_page_url": "https://doi.org/10.1234/test", "source": {"display_name": "Test Venue"}},
                    "best_oa_location": {"pdf_url": "https://example.org/paper.pdf", "landing_page_url": "https://example.org/paper", "is_oa": True, "version": "acceptedVersion"},
                    "abstract_inverted_index": {"Useful": [0], "abstract": [1]},
                    "cited_by_count": 7,
                    "referenced_works": ["https://openalex.org/W100"],
                    "related_works": ["https://openalex.org/W200"],
                    "topics": [{"display_name": "Machine learning"}],
                }]}).encode()

        with patch("backend.integrations.urlopen", return_value=FakeResponse()):
            paper = OpenAlexClient().resolve_paper("https://doi.org/10.1234/test")
        self.assertEqual(paper["openalexId"], "https://openalex.org/W123")
        self.assertEqual(paper["abstract"], "Useful abstract")
        self.assertEqual(paper["metadataSource"], "OpenAlex")
        self.assertEqual(paper["openAccessPdfUrl"], "https://example.org/paper.pdf")
        self.assertEqual(paper["openAccessVersion"], "acceptedVersion")

    def test_open_access_fulltext_is_verified_and_labeled(self):
        seed = {
            "title": "Reliability Aware Adaptive Self Consistency for Efficient Sampling in LLM",
            "doi": "10.1234/reliability",
            "sourceUrl": "https://doi.org/10.1234/reliability",
            "openAccessPdfUrl": "https://example.org/reliability.pdf",
            "openAccessLandingUrl": "https://example.org/reliability",
            "openAccessVersion": "submittedVersion",
        }

        class FakeSearch:
            def resolve_paper(self, query):
                return seed

            def rate_limit_status(self):
                return {"mode": "anonymous", "remaining": 90}

        bundle = ProviderBundle(paper_search=FakeSearch())
        extracted = {
            "text": "Reliability Aware Adaptive Self Consistency for Efficient Sampling in LLM\n" + "method and experiments " * 20,
            "pages": 9,
            "characters": 500,
            "truncated": False,
            "sections": ["方法", "实验"],
            "engine": "pdftotext",
        }
        with patch("backend.orchestrator.build_provider_bundle", return_value=bundle), patch(
            "backend.orchestrator.download_open_access_pdf",
            return_value={"bytes": b"%PDF-1.7\nmock", "url": seed["openAccessPdfUrl"], "contentType": "application/pdf"},
        ), patch("backend.orchestrator.extract_pdf_text", return_value=extracted):
            result = fetch_open_access_fulltext("10.1234/reliability", Settings.from_environment())
        self.assertTrue(result["ok"])
        self.assertEqual(result["source"]["kind"], "open-access-pdf")
        self.assertEqual(result["source"]["version"], "submittedVersion")
        self.assertTrue(result["source"]["identity"]["matched"])

    def test_fulltext_identity_handles_wrapped_doi_and_unicode_title(self):
        doi_match = _verify_fulltext_identity(
            "Article metadata DOI: 10.1234/\nreliability and publication details.",
            {"title": "Unrelated title", "doi": "10.1234/reliability"},
        )
        title_match = _verify_fulltext_identity(
            "可靠性感知的\n自适应自一致性方法\n作者：测试",
            {"title": "可靠性感知的自适应自一致性方法", "doi": ""},
        )
        mismatch = _verify_fulltext_identity(
            "An entirely different article about database indexing and storage engines.",
            {"title": "Reliability Aware Adaptive Self Consistency", "doi": "10.1234/reliability"},
        )
        self.assertEqual(doi_match["method"], "doi")
        self.assertEqual(title_match["method"], "title-exact")
        self.assertFalse(mismatch["matched"])

    def test_deep_abstract_fallback_is_visible_in_report_metadata_and_sources(self):
        seed = {
            "id": "w1", "openalexId": "https://openalex.org/W1", "doi": "10.1/seed",
            "title": "Seed Paper", "shortTitle": "Seed Paper", "authors": "A. Author",
            "year": 2026, "venue": "Test Venue", "identifier": "DOI · 10.1/seed",
            "sourceUrl": "https://doi.org/10.1/seed", "abstract": "A public abstract.",
            "tags": [], "citedByCount": 0, "metadataSource": "OpenAlex",
        }
        report = _build_live_report(
            "10.1/seed", seed, [], "", "deep",
            {"kind": "abstract-fallback", "label": "开放全文不可用", "reason": "OpenAlex 未收录开放 PDF"},
        )
        self.assertEqual(report["meta"]["version"], "0.7.4")
        self.assertTrue(report["meta"]["materialQuality"]["degraded"])
        self.assertEqual(report["meta"]["materialQuality"]["level"], "abstract-only")
        self.assertIn("材料降级", report["meta"]["materialScope"])
        self.assertEqual(report["provenance"][0]["kind"], "材料边界")
        self.assertIn("未收录开放 PDF", report["provenance"][0]["status"])

    def test_openalex_uses_method_acronym_as_short_title(self):
        raw = {
            "id": "https://openalex.org/W999",
            "title": "Dynamic Multi-Path Retrieval for Knowledge-based Visual Question Answering",
            "publication_year": 2026,
            "authorships": [],
            "primary_location": {"source": {}},
            "abstract_inverted_index": {
                "Knowledge-based": [0], "Visual": [1], "Question": [2], "Answering": [3], "(KB-VQA)": [4],
                "uses": [5], "Dynamic": [6], "Multi-Path": [7], "Retrieval": [8], "(DMRAG)": [9],
            },
            "referenced_works": [], "related_works": [], "topics": [],
        }
        client = OpenAlexClient()
        with patch.object(client, "_get", return_value={"results": [raw]}):
            paper = client.resolve_paper(raw["title"])
        self.assertEqual(paper["shortTitle"], "DMRAG")

    def test_short_title_prefers_acronym_whose_expansion_matches_title(self):
        raw = {
            "id": "https://openalex.org/W998",
            "title": "Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering",
            "publication_year": 2023,
            "authorships": [],
            "primary_location": {"source": {}},
            "abstract_inverted_index": {
                "Dense": [0], "Passage": [1], "Retrieval": [2, 9], "(DPR)": [3], "is": [4], "limited": [5],
                "Fine-grained": [6], "Late-interaction": [7], "Multi-modal": [8], "(FLMR)": [10],
                "improves": [11], "retrieval": [12],
            },
            "referenced_works": [], "related_works": [], "topics": [],
        }
        client = OpenAlexClient()
        with patch.object(client, "_get", return_value={"results": [raw]}):
            paper = client.resolve_paper(raw["title"])
        self.assertEqual(paper["shortTitle"], "FLMR")

    def test_sparse_openalex_record_triggers_one_semantic_fallback(self):
        seed = {
            "id": "w1",
            "openalexId": "https://openalex.org/W1",
            "title": "Sparse Seed Paper",
            "abstract": "Knowledge based visual question answering with external retrieval.",
            "relatedWorks": [],
        }
        raw_duplicate = {
            "id": "https://openalex.org/W99",
            "doi": "https://doi.org/10.1234/version",
            "title": "Sparse Seed Paper",
            "publication_year": 2026,
            "authorships": [],
            "primary_location": {"source": {}},
            "abstract_inverted_index": {"Duplicate": [0]},
            "cited_by_count": 0,
            "referenced_works": [],
            "related_works": [],
            "topics": [],
        }
        raw_candidate = {
            "id": "https://openalex.org/W2",
            "doi": "https://doi.org/10.1234/related",
            "title": "A Related Retrieval Paper",
            "publication_year": 2024,
            "authorships": [],
            "primary_location": {"source": {"display_name": "Test Venue"}},
            "abstract_inverted_index": {"Related": [0], "work": [1]},
            "cited_by_count": 2,
            "referenced_works": [],
            "related_works": [],
            "topics": [],
        }
        client = OpenAlexClient()
        with patch.object(client, "_get", return_value={"results": [raw_duplicate, raw_candidate]}) as mocked:
            papers = client.search_related(seed, [])
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(mocked.call_args.args[1]["per-page"], 10)
        self.assertIn("search.semantic", mocked.call_args.args[1])
        self.assertEqual(papers[0]["relation"], "语义检索")
        self.assertEqual(len(papers), 1)

    def test_non_sparse_related_works_do_not_trigger_semantic_fallback(self):
        seed = {
            "id": "w1",
            "openalexId": "https://openalex.org/W1",
            "title": "Well Connected Seed",
            "abstract": "Abstract text",
            "relatedWorks": ["https://openalex.org/W2", "https://openalex.org/W3", "https://openalex.org/W4"],
        }
        raw_results = []
        for number in (2, 3, 4):
            raw_results.append({
                "id": f"https://openalex.org/W{number}",
                "title": f"Related Paper {number}",
                "publication_year": 2023,
                "authorships": [],
                "primary_location": {"source": {}},
                "referenced_works": [],
                "related_works": [],
                "topics": [],
            })
        client = OpenAlexClient()
        with patch.object(client, "_get", return_value={"results": raw_results}) as mocked:
            papers = client.search_related(seed, [])
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(len(papers), 3)
        self.assertTrue(all(item["relation"] == "语义相关" for item in papers))

    def test_live_run_builds_arbitrary_paper_report(self):
        patch_payload = {
            "oneSentence": "模型基于实时元数据生成的一句话总结。",
            "problem": "模型基于公开摘要总结的研究问题描述。",
            "contributions": ["模型总结的第一项贡献", "模型总结的第二项贡献", "模型总结的第三项贡献"],
            "methodSteps": [
                {"label": "输入", "text": "模型总结的输入处理步骤"},
                {"label": "方法", "text": "模型总结的核心方法步骤"},
                {"label": "验证", "text": "模型总结的实验验证步骤"},
            ],
            "evidence": ["模型总结的第一项证据", "模型总结的第二项证据", "模型总结的第三项证据"],
            "limitations": ["模型总结的第一项局限", "模型总结的第二项局限", "模型总结的第三项局限"],
            "thesis": "模型基于相关工作形成的综合判断。",
            "gaps": ["模型识别的空白一", "模型识别的空白二", "模型识别的空白三", "模型识别的空白四"],
            "_provider": {
                "name": "zhipu", "label": "智谱 AI", "model": "glm-test",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "responseId": "rsp-live", "requestId": "req-live", "providerCreatedAt": 1234567890,
                "validationRetries": 0,
                "calls": [{"sequence": 1, "model": "glm-test", "responseId": "rsp-live", "requestId": "req-live", "providerCreatedAt": 1234567890, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}],
            },
            "deepDive": {
                "keyConcepts": [{"term": "动态检索", "explanation": "依据问题选择互补的多模态知识来源。"}],
                "technicalDetails": ["模型总结的技术机制细节"],
                "experimentalSetup": ["模型总结的实验设置"],
                "findings": ["模型总结的主要发现"],
                "readingGuide": ["模型生成的精读顺序"],
            },
        }

        seed = {
            "id": "w1", "openalexId": "https://openalex.org/W1", "doi": "10.1/seed", "title": "Seed Paper",
            "shortTitle": "Seed Paper", "authors": "A. Author", "year": 2024, "venue": "Venue", "identifier": "DOI · 10.1/seed",
            "sourceUrl": "https://doi.org/10.1/seed", "abstract": "The paper studies a well defined research problem.", "tags": ["AI"],
            "citedByCount": 3, "metadataSource": "OpenAlex",
        }
        related = dict(seed, id="w2", openalexId="https://openalex.org/W2", doi="10.1/related", title="Related Paper", shortTitle="Related Paper", relation="语义相关", why="推荐")
        reference = dict(seed, id="w3", openalexId="https://openalex.org/W3", doi="10.1/reference", title="Reference Paper", shortTitle="Reference Paper", relation="参考文献", why="引用")

        captured = {}
        progress_events = []

        class FakeLlm:
            def enhance_report(self, report):
                captured["source"] = report["_sourceMaterial"]
                return dict(patch_payload)

        class FakeSearch:
            def resolve_paper(self, query):
                return seed

            def search_related(self, paper, queries):
                return [related]

        class FakeCitation:
            def expand_citations(self, paper):
                return [reference]

        settings = Settings("zhipu", "temporary", "https://example.test/v4", "glm-test", "openalex-free", "", "openalex-free", "")
        bundle = ProviderBundle(llm=FakeLlm(), paper_search=FakeSearch(), citation_graph=FakeCitation())
        with patch("backend.orchestrator.build_provider_bundle", return_value=bundle):
            report = run_research(
                "A completely different paper",
                False,
                settings,
                source_text="METHOD 本文使用动态检索。 EXPERIMENT 在两个数据集上进行验证。",
                depth="deep",
                source_info={
                    "kind": "open-access-pdf",
                    "label": "Seed Paper",
                    "url": "https://example.test/seed.pdf",
                    "version": "acceptedVersion",
                },
                progress=lambda step_id, detail: progress_events.append((step_id, detail)),
            )
        self.assertEqual(report["meta"]["mode"], "live")
        self.assertEqual(report["paper"]["title"], "Seed Paper")
        self.assertEqual({item["relation"] for item in report["relatedPapers"]}, {"种子论文", "参考文献", "语义相关"})
        self.assertTrue(any(relation["type"] == "真实引用" for relation in report["relations"]))
        self.assertTrue(report["meta"]["modelCall"]["verified"])
        self.assertEqual(report["meta"]["modelCall"]["requestId"], "req-live")
        self.assertEqual(report["meta"]["modelCall"]["usage"]["total_tokens"], 15)
        self.assertEqual(report["meta"]["analysisDepth"], "deep")
        self.assertEqual(report["meta"]["materialScope"], "OpenAlex 开放 PDF（acceptedVersion）")
        self.assertEqual(report["provenance"][1]["kind"], "OpenAlex 定位的开放全文")
        self.assertEqual(report["provenance"][1]["url"], "https://example.test/seed.pdf")
        self.assertIn("仅在内存中参与分析", report["provenance"][1]["status"])
        self.assertIn("动态检索", captured["source"]["fullText"])
        self.assertNotIn("_sourceMaterial", report)
        self.assertEqual(report["analysis"]["deepDive"]["keyConcepts"][0]["term"], "动态检索")
        self.assertEqual(
            [item[0] for item in progress_events],
            ["resolve", "read", "discover", "verify", "analyze", "synthesize"],
        )
        self.assertNotIn("temporary", repr(report))


if __name__ == "__main__":
    unittest.main()

