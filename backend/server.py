"""Small dependency-free HTTP server for PaperScope."""

from __future__ import annotations

import argparse
import base64
import binascii
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from threading import RLock, Thread
from time import time
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .config import (
    LLM_PROVIDER_PRESETS,
    SUPPORTED_LLM_PROVIDERS,
    Settings,
    activate_runtime_llm,
    activate_runtime_openalex,
    deactivate_runtime_llm,
    deactivate_runtime_openalex,
    update_runtime_openalex_status,
)
from .integrations import IntegrationNotConfigured, ProviderRequestError
from .orchestrator import fetch_open_access_fulltext, get_pipeline, run_research, test_data_connection, test_llm_connection
from .pdf_text import extract_pdf_text
from .run_archive import get_run_archive
from .stage_cache import get_stage_cache


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "web"
_RESEARCH_JOBS: dict[str, dict] = {}
_RESEARCH_JOBS_LOCK = RLock()
_RESEARCH_JOB_TTL_SECONDS = 60 * 60


class PaperScopeHandler(BaseHTTPRequestHandler):
    server_version = "PaperScope/0.7.4"

    def log_message(self, format_string: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._json({"ok": True, "service": "PaperScope", "version": "0.7.4"})
            return
        if path == "/api/config":
            self._json(Settings.from_environment().public_status())
            return
        if path == "/api/pipeline":
            self._json({"steps": get_pipeline()})
            return
        if path == "/api/cache":
            self._json(get_stage_cache().status())
            return
        if path == "/api/research/runs":
            self._json({"runs": get_run_archive().list_recent()})
            return
        if path == "/api/research/status":
            job_id = str(parse_qs(parsed.query).get("jobId", [""])[0]).strip()
            job = _research_job_snapshot(job_id)
            if job is None:
                self._json({"error": "研究任务不存在或已过期。"}, HTTPStatus.NOT_FOUND)
            else:
                self._json(job)
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/research/start":
            try:
                payload = self._read_json()
                job = _start_research_job(payload, Settings.from_environment())
                self._json(job, HTTPStatus.ACCEPTED)
            except ValueError as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/extract/pdf":
            try:
                payload = self._read_json(max_bytes=17_000_000)
                encoded = str(payload.get("data", ""))
                pdf_bytes = base64.b64decode(encoded, validate=True)
                self._json({"ok": True, **extract_pdf_text(pdf_bytes)})
            except (ValueError, binascii.Error) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except NotImplementedError as error:
                self._json({"error": str(error), "code": "PDF_EXTRACTOR_UNAVAILABLE"}, HTTPStatus.NOT_IMPLEMENTED)
            return
        if path == "/api/fulltext/open-access":
            try:
                payload = self._read_json()
                self._json(fetch_open_access_fulltext(str(payload.get("query", "")), Settings.from_environment()))
            except ValueError as error:
                self._json({"error": str(error), "code": "OPEN_ACCESS_FULLTEXT_UNAVAILABLE"}, HTTPStatus.UNPROCESSABLE_ENTITY)
            except IntegrationNotConfigured as error:
                self._json({"error": str(error), "code": "INTEGRATION_NOT_CONFIGURED"}, HTTPStatus.SERVICE_UNAVAILABLE)
            except ProviderRequestError as error:
                self._json({"error": str(error), "code": "FULLTEXT_LOOKUP_FAILED"}, HTTPStatus.BAD_GATEWAY)
            except NotImplementedError as error:
                self._json({"error": str(error), "code": "PDF_EXTRACTOR_UNAVAILABLE"}, HTTPStatus.NOT_IMPLEMENTED)
            return
        if path == "/api/config/llm":
            try:
                candidate = _llm_candidate(self._read_json())
                connection = test_llm_connection(candidate)
                if not connection.get("ok"):
                    raise ProviderRequestError("模型已响应，但连通性校验未通过。")
                activate_runtime_llm(candidate)
                self._json({
                    "ok": True,
                    "connection": connection,
                    "config": Settings.from_environment().public_status(),
                })
            except ValueError as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except IntegrationNotConfigured as error:
                self._json({"error": str(error), "code": "PROVIDER_ACTIVATION_FAILED"}, HTTPStatus.BAD_GATEWAY)
            except ProviderRequestError as error:
                self._json({
                    "error": str(error),
                    "code": "PROVIDER_ACTIVATION_FAILED",
                    "failure": error.public_details(),
                }, HTTPStatus.BAD_GATEWAY)
            return
        if path == "/api/config/openalex":
            try:
                candidate = _openalex_candidate(self._read_json())
                connection = test_data_connection(candidate)
                if not connection.get("ok"):
                    raise ProviderRequestError("OpenAlex 已响应，但 API Key 校验未通过。")
                activate_runtime_openalex(candidate.paper_search_api_key, connection.get("rateLimit"))
                self._json({
                    "ok": True,
                    "connection": connection,
                    "config": Settings.from_environment().public_status(),
                })
            except ValueError as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except (IntegrationNotConfigured, ProviderRequestError) as error:
                self._json({"error": str(error), "code": "OPENALEX_ACTIVATION_FAILED"}, HTTPStatus.BAD_GATEWAY)
            return
        if path == "/api/llm/test":
            try:
                self._json(test_llm_connection(Settings.from_environment()))
            except IntegrationNotConfigured as error:
                self._json({"error": str(error), "code": "INTEGRATION_NOT_CONFIGURED"}, HTTPStatus.SERVICE_UNAVAILABLE)
            except ProviderRequestError as error:
                self._json({
                    "error": str(error),
                    "code": "PROVIDER_REQUEST_FAILED",
                    "failure": error.public_details(),
                }, HTTPStatus.BAD_GATEWAY)
            return
        if path == "/api/data/test":
            try:
                result = test_data_connection(Settings.from_environment())
                update_runtime_openalex_status(result.get("rateLimit"))
                self._json(result)
            except IntegrationNotConfigured as error:
                self._json({"error": str(error), "code": "INTEGRATION_NOT_CONFIGURED"}, HTTPStatus.SERVICE_UNAVAILABLE)
            except ProviderRequestError as error:
                self._json({"error": str(error), "code": "PROVIDER_REQUEST_FAILED"}, HTTPStatus.BAD_GATEWAY)
            return
        if path != "/api/research":
            self._json({"error": "未找到该 API。"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            started_at = time()
            report = run_research(
                query=str(payload.get("query", "")),
                demo=bool(payload.get("demo", False)),
                settings=Settings.from_environment(),
                source_text=str(payload.get("sourceText", "")),
                depth=str(payload.get("depth", "standard")),
                source_info=payload.get("sourceInfo"),
            )
            _archive_completed_report(report, f"sync-{uuid4().hex}", started_at)
            self._json(report)
        except ValueError as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except IntegrationNotConfigured as error:
            self._json({"error": str(error), "code": "INTEGRATION_NOT_CONFIGURED"}, HTTPStatus.SERVICE_UNAVAILABLE)
        except ProviderRequestError as error:
            self._json({
                "error": str(error),
                "code": "PROVIDER_REQUEST_FAILED",
                "failure": error.public_details(),
            }, HTTPStatus.BAD_GATEWAY)
        except NotImplementedError as error:
            self._json({"error": str(error), "code": "ADAPTER_NOT_IMPLEMENTED"}, HTTPStatus.NOT_IMPLEMENTED)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json({"error": "请求 JSON 无效。"}, HTTPStatus.BAD_REQUEST)

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/cache":
            self._json({"ok": True, **get_stage_cache().clear()})
            return
        if path == "/api/config/openalex":
            deactivate_runtime_openalex()
            self._json({"ok": True, "config": Settings.from_environment().public_status()})
            return
        if path != "/api/config/llm":
            self._json({"error": "未找到该 API。"}, HTTPStatus.NOT_FOUND)
            return
        deactivate_runtime_llm()
        self._json({"ok": True, "config": Settings.from_environment().public_status()})

    def _read_json(self, max_bytes: int = 500_000) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > max_bytes:
            raise ValueError("请求体过大。")
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("请求 JSON 必须是对象。")
        return payload

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaperScope locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), PaperScopeHandler)
    print(f"PaperScope is running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _start_research_job(payload: dict, settings: Settings) -> dict:
    query = str(payload.get("query", ""))
    if not query.strip():
        raise ValueError("请输入论文标题、DOI、arXiv 编号或论文链接。")
    job_id = uuid4().hex
    now = time()
    job = {
        "jobId": job_id,
        "status": "queued",
        "stepId": "resolve",
        "detail": "研究任务已创建，正在准备",
        "sequence": 0,
        "createdAt": now,
        "updatedAt": now,
    }
    with _RESEARCH_JOBS_LOCK:
        _purge_research_jobs(now)
        _RESEARCH_JOBS[job_id] = job
    thread = Thread(
        target=_run_research_job,
        args=(job_id, dict(payload), settings),
        name=f"paperscope-{job_id[:8]}",
        daemon=True,
    )
    thread.start()
    return _research_job_snapshot(job_id) or job


def _run_research_job(job_id: str, payload: dict, settings: Settings) -> None:
    def progress(step_id: str, detail: str) -> None:
        _update_research_job(job_id, status="running", stepId=step_id, detail=detail)

    try:
        progress("resolve", "正在启动研究流程")
        report = run_research(
            query=str(payload.get("query", "")),
            demo=bool(payload.get("demo", False)),
            settings=settings,
            source_text=str(payload.get("sourceText", "")),
            depth=str(payload.get("depth", "standard")),
            source_info=payload.get("sourceInfo"),
            progress=progress,
        )
        with _RESEARCH_JOBS_LOCK:
            started_at = float(_RESEARCH_JOBS.get(job_id, {}).get("createdAt", time()))
        _archive_completed_report(report, job_id, started_at)
        _update_research_job(
            job_id,
            status="completed",
            stepId="synthesize",
            detail="研究报告已生成",
            report=report,
        )
    except ValueError as error:
        _fail_research_job(job_id, str(error), "INVALID_REQUEST")
    except IntegrationNotConfigured as error:
        _fail_research_job(job_id, str(error), "INTEGRATION_NOT_CONFIGURED")
    except ProviderRequestError as error:
        _fail_research_job(
            job_id,
            str(error),
            "PROVIDER_REQUEST_FAILED",
            failure=error.public_details(),
        )
    except NotImplementedError as error:
        _fail_research_job(job_id, str(error), "ADAPTER_NOT_IMPLEMENTED")
    except Exception:
        _fail_research_job(job_id, "研究流程发生未预期错误，请重试。", "INTERNAL_ERROR")


def _archive_completed_report(report: dict, job_id: str, started_at: float) -> None:
    elapsed_seconds = round(max(0.0, time() - float(started_at)), 3)
    meta = report.setdefault("meta", {})
    meta["elapsedSeconds"] = elapsed_seconds
    if meta.get("mode") == "demo":
        return
    try:
        meta["runArchive"] = get_run_archive().save_completed(
            report,
            job_id=job_id,
            elapsed_seconds=elapsed_seconds,
        )
    except OSError:
        meta["runArchive"] = {"saved": False}


def _fail_research_job(job_id: str, message: str, code: str, failure: dict | None = None) -> None:
    category = str((failure or {}).get("category") or "")
    label = {
        "authentication": "模型认证失败",
        "permission": "模型无访问权限",
        "quota": "模型额度不足",
        "congestion": "模型服务拥堵",
        "service": "模型服务暂时异常",
        "network": "模型网络连接失败",
        "timeout": "模型响应超时",
        "invalid-response": "模型响应格式无效",
    }.get(category, "研究任务失败")
    _update_research_job(
        job_id,
        status="failed",
        detail=label,
        error=message,
        code=code,
        failure=failure or {},
        canResume=code == "PROVIDER_REQUEST_FAILED",
    )


def _update_research_job(job_id: str, **changes) -> None:
    with _RESEARCH_JOBS_LOCK:
        job = _RESEARCH_JOBS.get(job_id)
        if job is None:
            return
        job.update(changes)
        job["sequence"] = int(job.get("sequence", 0)) + 1
        job["updatedAt"] = time()


def _research_job_snapshot(job_id: str) -> dict | None:
    if not job_id:
        return None
    with _RESEARCH_JOBS_LOCK:
        job = _RESEARCH_JOBS.get(job_id)
        return dict(job) if job is not None else None


def _purge_research_jobs(now: float) -> None:
    expired = [
        job_id for job_id, job in _RESEARCH_JOBS.items()
        if now - float(job.get("updatedAt", now)) > _RESEARCH_JOB_TTL_SECONDS
    ]
    for job_id in expired:
        _RESEARCH_JOBS.pop(job_id, None)
    if len(_RESEARCH_JOBS) >= 64:
        oldest = min(_RESEARCH_JOBS, key=lambda item: float(_RESEARCH_JOBS[item].get("updatedAt", now)))
        _RESEARCH_JOBS.pop(oldest, None)


def _llm_candidate(payload: dict) -> Settings:
    provider = str(payload.get("provider", "")).strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError("请选择受支持的模型供应商。")
    preset = next(item for item in LLM_PROVIDER_PRESETS if item["id"] == provider)
    api_key = str(payload.get("apiKey", "")).strip()
    base_url = str(payload.get("baseUrl", "")).strip() or preset["baseUrl"]
    model = str(payload.get("model", "")).strip() or preset["defaultModel"]
    if len(api_key) < 8:
        raise ValueError("请输入有效的 API Key。")
    if not base_url or len(base_url) > 1000:
        raise ValueError("请输入有效的 Base URL。")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Base URL 必须是有效的 HTTP(S) 地址，且不能包含账号信息。")
    if not model or len(model) > 200:
        raise ValueError("请输入有效的模型名称。")
    return Settings.from_environment().with_llm(provider, api_key, base_url, model)


def _openalex_candidate(payload: dict) -> Settings:
    api_key = str(payload.get("apiKey", "")).strip()
    if len(api_key) < 8 or len(api_key) > 500:
        raise ValueError("请输入有效的 OpenAlex API Key。")
    return Settings.from_environment().with_openalex_api_key(api_key)


if __name__ == "__main__":
    main()

