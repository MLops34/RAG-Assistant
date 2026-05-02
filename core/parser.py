"""
Syllabus parsing: extract text from PDFs and optionally structure with an LLM.

Requires ``OPENAI_API_KEY`` in the environment for structured LLM parsing.
Without a key, ``parse_syllabus_pdf(..., use_llm=False)`` returns raw text only.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Union

# Project root on path for `models` when not installed as a package
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pypdf import PdfReader

from core.llm_provider import get_llm_provider_config
from models.syllabus import ParsedSyllabus

PathLike = Union[str, Path]

# Rough token-safe cap for LLM context (characters); adjust per model
_DEFAULT_MAX_CHARS_FOR_LLM = 48_000


def _normalize_extracted_text(text: str) -> str:
    """Normalize noisy PDF extraction into a parser-friendly text block."""
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove isolated page numbers and common running headers/footers noise.
    cleaned_lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.fullmatch(r"(page\s*)?\d{1,3}(\s*of\s*\d{1,3})?", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    normalized = "\n".join(cleaned_lines)
    # Fix PDF hyphenation line-break artifacts.
    normalized = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", normalized)
    # Remove repeated excessive blank lines after cleaning.
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


def _text_quality_is_low(text: str) -> bool:
    """Heuristic to detect low-quality extraction likely needing fallback strategy."""
    if not text.strip():
        return True
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(words) < 80:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    alpha_chars = sum(ch.isalpha() for ch in text)
    total_chars = max(1, len(text))
    alpha_ratio = alpha_chars / total_chars
    short_line_ratio = (
        sum(1 for line in lines if len(line.strip()) <= 2) / max(1, len(lines))
    )
    return alpha_ratio < 0.45 or short_line_ratio > 0.12


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Chunk text by paragraph boundaries to preserve topic context."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        extra = len(para) + (2 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += extra
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _merge_unique_topics(base: ParsedSyllabus, incoming: ParsedSyllabus) -> None:
    seen: set[str] = {t.title.strip().lower() for t in base.topics if t.title.strip()}
    for topic in incoming.topics:
        key = topic.title.strip().lower()
        if key and key not in seen:
            base.topics.append(topic)
            seen.add(key)
    existing_exam_keys = {
        (e.name.strip().lower(), e.date.isoformat() if e.date else "")
        for e in base.exam_dates
    }
    for exam in incoming.exam_dates:
        exam_key = (exam.name.strip().lower(), exam.date.isoformat() if exam.date else "")
        if exam_key not in existing_exam_keys:
            base.exam_dates.append(exam)
            existing_exam_keys.add(exam_key)


def extract_text_from_pdf(path: PathLike) -> str:
    """Extract plain text from a PDF file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return _normalize_extracted_text("\n\n".join(parts).strip())


def _parse_with_llm_single(text: str, max_chars: int = _DEFAULT_MAX_CHARS_FOR_LLM) -> ParsedSyllabus:
    """Use OpenAI via LangChain structured output to fill ``ParsedSyllabus``."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    provider = get_llm_provider_config()

    truncated = text[:max_chars]
    if len(text) > max_chars:
        truncated += "\n\n[... document truncated for model context ...]"

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_SYLLABUS_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=provider.api_key,
        base_url=provider.base_url,
    )
    structured = llm.with_structured_output(ParsedSyllabus)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract structured data from university course syllabi. "
                "Infer exam dates only when explicitly stated; otherwise leave date null. "
                "Topics should reflect major units or modules, in syllabus order when possible. "
                "Return concise, clean topic names without duplicates. "
                "Leave raw_text empty; the application fills it after parsing.",
            ),
            (
                "human",
                "Parse this syllabus text into the required schema.\n\n{syllabus}\n\n"
                "If text seems partial, still extract whatever topic and exam signals exist.",
            ),
        ]
    )
    chain = prompt | structured
    result: ParsedSyllabus = chain.invoke({"syllabus": truncated})
    return result


def _parse_with_llm(text: str, max_chars: int = _DEFAULT_MAX_CHARS_FOR_LLM) -> ParsedSyllabus:
    """Chunk-aware LLM parsing for better large/noisy syllabus coverage."""
    chunks = _chunk_text(text, max_chars=max_chars)
    if len(chunks) == 1:
        return _parse_with_llm_single(chunks[0], max_chars=max_chars)

    merged = ParsedSyllabus()
    # Parse at most first 3 chunks to balance quality, latency and cost.
    for chunk in chunks[:3]:
        parsed_chunk = _parse_with_llm_single(chunk, max_chars=max_chars)
        _merge_unique_topics(merged, parsed_chunk)
    return merged


def parse_syllabus_pdf(
    path: PathLike,
    *,
    use_llm: bool = True,
    max_chars_for_llm: int = _DEFAULT_MAX_CHARS_FOR_LLM,
) -> ParsedSyllabus:
    """
    Load a syllabus PDF, extract text, and optionally fill structured fields with an LLM.

    If ``use_llm`` is False, returns a ``ParsedSyllabus`` with ``raw_text`` set and
    empty topics/exams (suitable for RAG-only flows without API cost).
    """
    path = Path(path)
    raw = extract_text_from_pdf(path)

    if not use_llm:
        return ParsedSyllabus(raw_text=raw, source_path=str(path.resolve()))

    if _text_quality_is_low(raw):
        # Extraction is weak; still proceed, but parsed structure may be sparse.
        # The caller UI surfaces quality hints and fallback enrichment.
        pass
    parsed = _parse_with_llm(raw, max_chars=max_chars_for_llm)
    parsed.raw_text = raw
    parsed.source_path = str(path.resolve())
    return parsed


def save_parsed_syllabus(data: ParsedSyllabus, json_path: PathLike) -> None:
    """Write ``ParsedSyllabus`` to JSON (excludes huge raw_text if you strip it first)."""
    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(data.model_dump_json(indent=2), encoding="utf-8")


def load_parsed_syllabus(json_path: PathLike) -> ParsedSyllabus:
    """Load ``ParsedSyllabus`` from JSON."""
    path = Path(json_path)
    return ParsedSyllabus.model_validate_json(path.read_text(encoding="utf-8"))


def parse_syllabus_from_text(text: str, *, use_llm: bool = True) -> ParsedSyllabus:
    """Parse already-extracted syllabus text (e.g. from OCR or paste)."""
    text = _normalize_extracted_text(text.strip())
    if not use_llm:
        return ParsedSyllabus(raw_text=text)
    parsed = _parse_with_llm(text)
    parsed.raw_text = text
    return parsed
