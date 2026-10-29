"""Local PDF text extraction without persisting uploaded documents."""

from __future__ import annotations

import ipaddress
import re
from shutil import which
from subprocess import PIPE, run
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MAX_PDF_BYTES = 12 * 1024 * 1024
MAX_EXTRACTED_CHARACTERS = 80_000


def download_open_access_pdf(url: str, timeout: float = 30.0) -> dict:
    """Download a trusted OA PDF into memory after basic SSRF and size checks."""
    _validate_public_https_url(url)
    request = Request(
        url,
        headers={
            "Accept": "application/pdf,application/octet-stream;q=0.8",
            "User-Agent": "PaperScope/0.7.4 (local research assistant)",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            final_url = str(response.geturl() or url)
            _validate_public_https_url(final_url)
            length = str(response.headers.get("Content-Length") or "").strip()
            if length.isdigit() and int(length) > MAX_PDF_BYTES:
                raise ValueError("开放全文 PDF 超过 12 MB，未自动下载。")
            content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            pdf_bytes = response.read(MAX_PDF_BYTES + 1)
    except HTTPError as error:
        raise ValueError(f"开放全文下载失败（HTTP {error.code}）。") from error
    except URLError as error:
        raise ValueError("无法连接开放全文来源，请稍后重试。") from error
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("开放全文 PDF 超过 12 MB，未自动下载。")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("开放全文链接没有返回有效 PDF，可能需要登录或已失效。")
    return {"bytes": pdf_bytes, "url": final_url, "contentType": content_type}


def _validate_public_https_url(value: str) -> None:
    parsed = urlparse(str(value or "").strip())
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
        raise ValueError("开放全文只允许从公开 HTTPS 链接读取。")
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("开放全文链接指向本地地址，已拒绝读取。")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("开放全文链接指向非公开网络地址，已拒绝读取。")


def extract_pdf_text(pdf_bytes: bytes) -> dict:
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("文件不是有效的 PDF。")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF 最大支持 12 MB，请只保留论文正文页面。")
    executable = which("pdftotext")
    if not executable:
        raise NotImplementedError("当前电脑未安装 pdftotext，暂时请从 PDF 复制正文后粘贴。")
    result = run(
        [executable, "-layout", "-enc", "UTF-8", "-", "-"],
        input=pdf_bytes,
        stdout=PIPE,
        stderr=PIPE,
        timeout=45,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()[:300]
        raise ValueError(f"PDF 文本提取失败：{message or '文件可能已加密或损坏'}")
    raw = result.stdout.decode("utf-8", errors="replace")
    page_count = max(1, raw.rstrip("\f\r\n ").count("\f") + 1)
    text = _clean_pdf_text(raw)
    if len(text) < 80:
        raise ValueError("PDF 中没有提取到足够文本；它可能是扫描件，需要 OCR 后再导入。")
    truncated = len(text) > MAX_EXTRACTED_CHARACTERS
    text = text[:MAX_EXTRACTED_CHARACTERS]
    return {
        "text": text,
        "pages": page_count,
        "characters": len(text),
        "truncated": truncated,
        "sections": _detect_sections(text),
        "engine": "pdftotext",
    }


def _clean_pdf_text(value: str) -> str:
    value = value.replace("\x00", "").replace("\f", "\n\n")
    lines = []
    blank = False
    for raw_line in value.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if not blank:
                lines.append("")
            blank = True
            continue
        blank = False
        lines.append(line)
    return "\n".join(lines).strip()


def _detect_sections(text: str) -> list[str]:
    patterns = (
        ("摘要", r"^(?:abstract|摘要)\b"),
        ("引言", r"^(?:\d+[.\s]+)?(?:introduction|引言|绪论)\b"),
        ("相关工作", r"^(?:\d+[.\s]+)?(?:related work|background|相关工作|研究背景)\b"),
        ("方法", r"^(?:\d+[.\s]+)?(?:method|methodology|approach|model|方法|模型)\b"),
        ("实验", r"^(?:\d+[.\s]+)?(?:experiments?|evaluation|results?|实验|评估|结果)\b"),
        ("局限", r"^(?:\d+[.\s]+)?(?:limitation|discussion|局限|讨论)\b"),
        ("结论", r"^(?:\d+[.\s]+)?(?:conclusion|结论)\b"),
        ("参考文献", r"^(?:references|bibliography|参考文献)\b"),
    )
    found = []
    for line in text.splitlines():
        probe = line.strip().lower()
        if not probe or len(probe) > 100:
            continue
        for label, pattern in patterns:
            if label not in found and re.search(pattern, probe, flags=re.IGNORECASE):
                found.append(label)
    return found
