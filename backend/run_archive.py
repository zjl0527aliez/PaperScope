"""Privacy-conscious archive for recent completed research reports.

Only the generated report is retained. Raw source text, uploaded PDF content,
API keys and request payloads are removed recursively before writing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4


ARCHIVE_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "runtime" / "research-runs"
_SENSITIVE_KEYS = {
    "apikey", "api_key", "authorization", "access_token", "refresh_token", "password",
    "sourcetext", "fulltext", "_sourcematerial",
}


class RunArchive:
    def __init__(self, root: Path = DEFAULT_ARCHIVE_ROOT, max_entries: int = 24):
        self.root = Path(root)
        self.max_entries = max(4, int(max_entries))
        self._lock = RLock()

    def save_completed(self, report: dict, *, job_id: str, elapsed_seconds: float) -> dict:
        archived_at = datetime.now(timezone.utc)
        archive_id = f"{archived_at.strftime('%Y%m%dT%H%M%S%fZ')}-{_safe_id(job_id)}"
        safe_report = _sanitize(report)
        payload = {
            "schemaVersion": ARCHIVE_SCHEMA_VERSION,
            "archiveId": archive_id,
            "jobId": str(job_id),
            "status": "completed",
            "archivedAt": archived_at.isoformat(),
            "elapsedSeconds": round(max(0.0, float(elapsed_seconds)), 3),
            "report": safe_report,
        }
        path = self.root / f"{archive_id}.json"
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            self._prune_locked()
        return {
            "archiveId": archive_id,
            "archivedAt": payload["archivedAt"],
            "elapsedSeconds": payload["elapsedSeconds"],
        }

    def list_recent(self) -> list[dict]:
        items = []
        with self._lock:
            paths = sorted(self.root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True) if self.root.is_dir() else []
            for path in paths:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
                meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
                paper = report.get("paper") if isinstance(report.get("paper"), dict) else {}
                usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
                model_call = meta.get("modelCall") if isinstance(meta.get("modelCall"), dict) else {}
                items.append({
                    "archiveId": payload.get("archiveId"),
                    "archivedAt": payload.get("archivedAt"),
                    "elapsedSeconds": payload.get("elapsedSeconds"),
                    "title": paper.get("title"),
                    "provider": meta.get("provider"),
                    "model": meta.get("model"),
                    "analysisDepth": meta.get("analysisDepth"),
                    "materialScope": meta.get("materialScope"),
                    "totalTokens": usage.get("total_tokens"),
                    "promptCacheHitTokens": usage.get("prompt_cache_hit_tokens"),
                    "promptCacheMissTokens": usage.get("prompt_cache_miss_tokens"),
                    "modelElapsedSeconds": model_call.get("modelElapsedSeconds"),
                })
        return items

    def _prune_locked(self) -> None:
        paths = sorted(self.root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for path in paths[self.max_entries:]:
            try:
                path.unlink()
            except OSError:
                continue


def _sanitize(value):
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if str(key).replace("-", "").lower() not in _SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _safe_id(value: str) -> str:
    result = "".join(character for character in str(value) if character.isalnum() or character in {"-", "_"})
    return (result or uuid4().hex)[:64]


_RUN_ARCHIVE = RunArchive()


def get_run_archive() -> RunArchive:
    return _RUN_ARCHIVE
