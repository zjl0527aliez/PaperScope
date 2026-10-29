"""Runtime configuration with process-memory API credentials."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from threading import RLock


LLM_PROVIDER_PRESETS = (
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "baseUrl": "https://api.deepseek.com",
        "defaultModel": "deepseek-flash",
    },
    {
        "id": "zhipu",
        "label": "智谱 AI",
        "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
        "defaultModel": "glm-4.7-flash",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "baseUrl": "https://api.openai.com/v1",
        "defaultModel": "",
    },
    {
        "id": "qwen",
        "label": "通义千问",
        "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "defaultModel": "qwen-plus",
    },
    {
        "id": "openai-compatible",
        "label": "其他 OpenAI 兼容接口",
        "baseUrl": "",
        "defaultModel": "",
    },
)

SUPPORTED_LLM_PROVIDERS = {item["id"] for item in LLM_PROVIDER_PRESETS}
_PROVIDER_LABELS = {item["id"]: item["label"] for item in LLM_PROVIDER_PRESETS}
_runtime_lock = RLock()
_runtime_llm_config: dict[str, str] | None = None
_runtime_openalex_key: str | None = None
_runtime_openalex_status: dict = {}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    paper_search_provider: str
    paper_search_api_key: str
    citation_provider: str
    citation_api_key: str

    @classmethod
    def from_environment(cls) -> "Settings":
        value = lambda name: os.environ.get(name, "").strip()
        common_openalex_key = value("PAPERSCOPE_OPENALEX_API_KEY")
        settings = cls(
            llm_provider=value("PAPERSCOPE_LLM_PROVIDER"),
            llm_api_key=value("PAPERSCOPE_LLM_API_KEY"),
            llm_base_url=value("PAPERSCOPE_LLM_BASE_URL"),
            llm_model=value("PAPERSCOPE_LLM_MODEL"),
            paper_search_provider=value("PAPERSCOPE_PAPER_SEARCH_PROVIDER") or "openalex-free",
            paper_search_api_key=value("PAPERSCOPE_PAPER_SEARCH_API_KEY") or common_openalex_key,
            citation_provider=value("PAPERSCOPE_CITATION_PROVIDER") or "openalex-free",
            citation_api_key=value("PAPERSCOPE_CITATION_API_KEY") or common_openalex_key,
        )
        with _runtime_lock:
            runtime = dict(_runtime_llm_config) if _runtime_llm_config is not None else None
            runtime_openalex = _runtime_openalex_key
        if runtime is not None:
            settings = settings.with_llm(
                provider=runtime.get("provider", ""),
                api_key=runtime.get("api_key", ""),
                base_url=runtime.get("base_url", ""),
                model=runtime.get("model", ""),
            )
        if runtime_openalex is not None:
            settings = settings.with_openalex_api_key(runtime_openalex)
        return settings

    def with_llm(self, provider: str, api_key: str, base_url: str, model: str) -> "Settings":
        return replace(
            self,
            llm_provider=provider.strip().lower(),
            llm_api_key=api_key.strip(),
            llm_base_url=base_url.strip().rstrip("/"),
            llm_model=model.strip(),
        )

    def with_openalex_api_key(self, api_key: str) -> "Settings":
        clean_key = api_key.strip()
        return replace(
            self,
            paper_search_api_key=clean_key,
            citation_api_key=clean_key,
        )

    def public_status(self) -> dict:
        """Return configuration state and presets without exposing secrets."""
        provider = self.llm_provider.lower()
        openalex_keyed = bool(self.paper_search_api_key or self.citation_api_key)
        openalex_status = get_runtime_openalex_status()
        data_common = {
            "configured": True,
            "provider": "openalex" if openalex_keyed else "openalex-free",
            "providerLabel": "OpenAlex 免费 API Key" if openalex_keyed else "OpenAlex 匿名模式",
            "mode": "api-key" if openalex_keyed else "anonymous",
            "cost": "每日免费 $1 预算" if openalex_keyed else "免费匿名额度",
            "storage": "process-memory" if openalex_keyed and _runtime_openalex_key is not None else None,
            "rateLimit": openalex_status or None,
        }
        return {
            "mode": "live" if self.live_ready else "assisted" if self.llm_ready else "demo",
            "llm": {
                "configured": self.llm_ready,
                "provider": provider or None,
                "providerLabel": provider_label(provider) if provider else None,
                "model": self.llm_model or None,
                "baseUrl": self.llm_base_url or None,
                "storage": "process-memory" if self.llm_ready else None,
            },
            "llmProviders": [dict(item) for item in LLM_PROVIDER_PRESETS],
            "paperSearch": {**data_common, "configured": bool(self.paper_search_provider)},
            "citationGraph": {**data_common, "configured": bool(self.citation_provider)},
        }

    @property
    def llm_ready(self) -> bool:
        return bool(
            self.llm_provider in SUPPORTED_LLM_PROVIDERS
            and self.llm_api_key
            and self.llm_base_url
            and self.llm_model
        )

    @property
    def live_ready(self) -> bool:
        return bool(self.llm_ready and self.paper_search_provider)


def provider_label(provider: str) -> str:
    return _PROVIDER_LABELS.get(provider.lower(), provider or "模型 API")


def _data_provider_label(provider: str) -> str | None:
    if not provider:
        return None
    if provider == "openalex-free":
        return "OpenAlex 免费模式"
    return provider


def activate_runtime_llm(settings: Settings) -> None:
    """Keep credentials in process memory only after a successful connection test."""
    global _runtime_llm_config
    with _runtime_lock:
        _runtime_llm_config = {
            "provider": settings.llm_provider,
            "api_key": settings.llm_api_key,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
        }


def deactivate_runtime_llm() -> None:
    """Mask environment credentials for this process until it restarts."""
    global _runtime_llm_config
    with _runtime_lock:
        _runtime_llm_config = {"provider": "", "api_key": "", "base_url": "", "model": ""}


def activate_runtime_openalex(api_key: str, status: dict | None = None) -> None:
    """Use one OpenAlex key for metadata search and citation expansion in memory."""
    global _runtime_openalex_key, _runtime_openalex_status
    with _runtime_lock:
        _runtime_openalex_key = api_key.strip()
        _runtime_openalex_status = dict(status or {})


def update_runtime_openalex_status(status: dict | None) -> None:
    global _runtime_openalex_status
    if not isinstance(status, dict):
        return
    with _runtime_lock:
        if _runtime_openalex_key:
            _runtime_openalex_status = {**_runtime_openalex_status, **status}


def deactivate_runtime_openalex() -> None:
    """Return to anonymous OpenAlex access until this process restarts."""
    global _runtime_openalex_key, _runtime_openalex_status
    with _runtime_lock:
        _runtime_openalex_key = ""
        _runtime_openalex_status = {}


def get_runtime_openalex_status() -> dict:
    with _runtime_lock:
        return dict(_runtime_openalex_status)


def reset_runtime_llm() -> None:
    """Return to environment-only configuration. Used by tests and advanced callers."""
    global _runtime_llm_config
    with _runtime_lock:
        _runtime_llm_config = None


def reset_runtime_openalex() -> None:
    """Return to environment-only OpenAlex configuration. Used by tests."""
    global _runtime_openalex_key, _runtime_openalex_status
    with _runtime_lock:
        _runtime_openalex_key = None
        _runtime_openalex_status = {}
