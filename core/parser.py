"""
Syllabus parsing: extract text from PDFs and optionally structure with an LLM.

Requires ``OPENAI_API_KEY`` in the environment for structured LLM parsing.
Without a key, ``parse_syllabus_pdf(..., use_llm=False)`` returns raw text only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union

# Project root on path for `models` when not installed as a package
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pypdf import PdfReader

from models.syllabus import ParsedSyllabus

PathLike = Union[str, Path]

# Rough token-safe cap for LLM context (characters); adjust per model
_DEFAULT_MAX_CHARS_FOR_LLM = 48_000


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
    return "\n\n".join(parts).strip()


def _parse_with_llm(text: str, max_chars: int = _DEFAULT_MAX_CHARS_FOR_LLM) -> ParsedSyllabus:
    """Use OpenAI via LangChain structured output to fill ``ParsedSyllabus``."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Export it or pass use_llm=False and use raw_text only."
        )

    truncated = text[:max_chars]
    if len(text) > max_chars:
        truncated += "\n\n[... document truncated for model context ...]"

    llm = ChatOpenAI(model=os.getenv("OPENAI_SYLLABUS_MODEL", "gpt-4o-mini"), temperature=0)
    structured = llm.with_structured_output(ParsedSyllabus)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract structured data from university course syllabi. "
                "Infer exam dates only when explicitly stated; otherwise leave date null. "
                "Topics should reflect major units or modules, in syllabus order when possible. "
                "Leave raw_text empty; the application fills it after parsing.",
            ),
            (
                "human",
                "Parse this syllabus text into the required schema.\n\n{syllabus}",
            ),
        ]
    )
    chain = prompt | structured
    result: ParsedSyllabus = chain.invoke({"syllabus": truncated})
    return result


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
    if not use_llm:
        return ParsedSyllabus(raw_text=text.strip())
    parsed = _parse_with_llm(text.strip())
    parsed.raw_text = text.strip()
    return parsed
