"""Validated local cache for model-stage outputs.

Cache files contain model outputs and non-secret request metadata only. API keys,
raw prompts, and source materials are never stored. The SHA-256 cache key binds
the provider, endpoint, model, prompt version, complete prompt content, and token
budget, so any meaningful input change produces a miss.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4


CACHE_SCHEMA_VERSION = 1
PROMPT_VERSION = "2026-10-29-v8"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "runtime" / "stage-cache"


class StageCache:
    def __init__(self, root: Path = DEFAULT_CACHE_ROOT, max_entries: int = 256):
        self.root = Path(root)
        self.max_entries = max(8, int(max_entries))
        self._lock = RLock()

    def make_key(
        self,
        *,
        provider: str,
        base_url: str,
        model: str,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        payload = {
            "schemaVersion": CACHE_SCHEMA_VERSION,
            "promptVersion": PROMPT_VERSION,
            "provider": provider,
            "baseUrl": base_url.rstrip("/"),
            "model": model,
            "stage": stage,
            "systemPrompt": system_prompt,
            "userPrompt": user_prompt,
            "maxTokens": int(max_tokens),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def load(self, stage: str, cache_key: str) -> dict | None:
        path = self._path(stage, cache_key)
        with self._lock:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        if not isinstance(payload, dict):
            return None
        if payload.get("schemaVersion") != CACHE_SCHEMA_VERSION:
            return None
        if payload.get("promptVersion") != PROMPT_VERSION:
            return None
        if payload.get("stage") != stage or payload.get("cacheKey") != cache_key:
            return None
        if not isinstance(payload.get("result"), dict):
            return None
        return payload

    def save(
        self,
        *,
        stage: str,
        cache_key: str,
        provider: str,
        model: str,
        result: dict,
        fallbacks: list[str],
    ) -> None:
        payload = {
            "schemaVersion": CACHE_SCHEMA_VERSION,
            "promptVersion": PROMPT_VERSION,
            "cacheKey": cache_key,
            "stage": stage,
            "provider": provider,
            "model": model,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "fallbacks": list(fallbacks),
        }
        path = self._path(stage, cache_key)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            self._prune_locked()

    def clear(self) -> dict:
        removed = 0
        freed_bytes = 0
        with self._lock:
            if self.root.is_dir():
                for path in self.root.glob("*.json"):
                    try:
                        freed_bytes += path.stat().st_size
                        path.unlink()
                        removed += 1
                    except OSError:
                        continue
        return {"removed": removed, "freedBytes": freed_bytes, **self.status()}

    def status(self) -> dict:
        entries = 0
        size_bytes = 0
        stale_entries = 0
        stale_size_bytes = 0
        stages: dict[str, int] = {}
        with self._lock:
            if self.root.is_dir():
                for path in self.root.glob("*.json"):
                    try:
                        file_size = path.stat().st_size
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        if (
                            not isinstance(payload, dict)
                            or payload.get("schemaVersion") != CACHE_SCHEMA_VERSION
                            or payload.get("promptVersion") != PROMPT_VERSION
                        ):
                            stale_entries += 1
                            stale_size_bytes += file_size
                            continue
                        size_bytes += file_size
                        entries += 1
                        stage = str(payload.get("stage") or "other")
                        stages[stage] = stages.get(stage, 0) + 1
                    except (OSError, json.JSONDecodeError):
                        stale_entries += 1
                        continue
        return {
            "enabled": True,
            "entries": entries,
            "sizeBytes": size_bytes,
            "staleEntries": stale_entries,
            "staleSizeBytes": stale_size_bytes,
            "maxEntries": self.max_entries,
            "promptVersion": PROMPT_VERSION,
            "stages": stages,
            "storesSecrets": False,
        }

    def _path(self, stage: str, cache_key: str) -> Path:
        safe_stage = "".join(character for character in stage if character.isalnum() or character in {"-", "_"})
        return self.root / f"{safe_stage}-{cache_key}.json"

    def _prune_locked(self) -> None:
        paths = []
        for path in self.root.glob("*.json"):
            try:
                paths.append((path.stat().st_mtime, path))
            except OSError:
                continue
        excess = len(paths) - self.max_entries
        for _, path in sorted(paths)[:max(0, excess)]:
            try:
                path.unlink()
            except OSError:
                continue


_STAGE_CACHE = StageCache()


def get_stage_cache() -> StageCache:
    return _STAGE_CACHE
