"""FastAPI layer for Next.js frontend integration."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import (
    _build_schedule_insights,
    _compute_parse_quality,
    ensure_topics_for_scheduling,
    optional_rag_query,
)
from core.llm_provider import get_llm_provider_config
from core.optimizer import DailyLimits, DeepWorkWindow, StudyScheduler
from core.parser import parse_syllabus_pdf
from langchain_openai import ChatOpenAI
from models.syllabus import ParsedSyllabus


class ParseResponse(BaseModel):
    syllabus: dict
    parse_quality: dict


class ScheduleTopicInput(BaseModel):
    title: str
    priority: float = 1.0
    target_minutes: int = 0
    difficulty: float = 1.0
    has_deadline: bool = False
    deadline: date | None = None


class ScheduleRequest(BaseModel):
    syllabus: dict
    topics: list[ScheduleTopicInput]
    optimizer_mode: str = Field(default="cp_sat", pattern="^(cp_sat|greedy)$")
    include_reviews: bool = True
    strict_mode: bool = True
    query: str | None = None
    no_study_weekdays: list[int] = Field(default_factory=list)


class ChatUpdate(BaseModel):
    title: str
    priority: float | None = None
    target_minutes: int | None = None
    difficulty: float | None = None
    has_deadline: bool | None = None
    deadline: date | None = None


class ChatAdjustRequest(BaseModel):
    syllabus: dict
    topics: list[ScheduleTopicInput]
    message: str


def _default_windows() -> list[DeepWorkWindow]:
    return [
        DeepWorkWindow(weekday=0, start_time=time(19, 0), end_time=time(21, 0)),
        DeepWorkWindow(weekday=2, start_time=time(19, 0), end_time=time(21, 0)),
        DeepWorkWindow(weekday=5, start_time=time(10, 0), end_time=time(12, 0)),
    ]


app = FastAPI(title="RAG Assistant API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/parse", response_model=ParseResponse)
async def parse_endpoint(
    file: UploadFile = File(...),
    use_llm: bool = Form(default=True),
) -> ParseResponse:
    suffix = Path(file.filename or "syllabus.pdf").suffix or ".pdf"
    temp_path = Path(f".upload_{datetime.now().timestamp()}{suffix}")
    try:
        payload = await file.read()
        temp_path.write_bytes(payload)
        parsed = parse_syllabus_pdf(str(temp_path), use_llm=use_llm)
        parsed = ensure_topics_for_scheduling(parsed)
        if not parsed.topics:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract usable topics. Try enabling LLM parsing "
                    "or upload a clearer text-selectable PDF."
                ),
            )
        quality = _compute_parse_quality(parsed.raw_text or "", len(parsed.topics))
        return ParseResponse(
            syllabus=parsed.model_dump(mode="json"),
            parse_quality=quality,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/schedule")
def schedule_endpoint(payload: ScheduleRequest) -> dict:
    try:
        syllabus = ParsedSyllabus.model_validate(payload.syllabus)
        topic_minutes_override: dict[str, int] = {}
        topic_difficulty: dict[str, float] = {}
        topic_deadlines: dict[str, date] = {}

        for item in payload.topics:
            for topic in syllabus.topics:
                if topic.title == item.title:
                    topic.weightage_percent = float(item.priority)
                    topic.estimated_hours = float(item.target_minutes) / 60 if item.target_minutes > 0 else None
                    break
            topic_minutes_override[item.title] = int(item.target_minutes)
            topic_difficulty[item.title] = float(item.difficulty)
            if item.has_deadline and item.deadline:
                topic_deadlines[item.title] = item.deadline

        rag_answer = optional_rag_query(syllabus, payload.query)
        scheduler = StudyScheduler(preferred_mode=payload.optimizer_mode)
        explicit_minutes_mode = any(v > 0 for v in topic_minutes_override.values())
        blocks = scheduler.build_schedule(
            syllabus=syllabus,
            deep_work_windows=_default_windows(),
            daily_limits=DailyLimits(),
            include_reviews=payload.include_reviews,
            topic_minutes_override=topic_minutes_override if explicit_minutes_mode else None,
            topic_difficulty=topic_difficulty,
            topic_deadlines=topic_deadlines,
            no_study_weekdays=set(payload.no_study_weekdays),
            strict_mode=payload.strict_mode,
        )
        analysis_rows = _build_schedule_insights(
            syllabus=syllabus,
            topic_minutes_override=topic_minutes_override,
            topic_difficulty=topic_difficulty,
            topic_deadlines=topic_deadlines,
            blocks=blocks,
        )
        return {
            "blocks": [asdict(b) for b in blocks],
            "analysis": analysis_rows,
            "rag_answer": rag_answer,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat-adjust")
def chat_adjust_endpoint(payload: ChatAdjustRequest) -> dict:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        provider = get_llm_provider_config()
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=provider.api_key,
            base_url=provider.base_url,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        topic_rows = [item.model_dump(mode="json") for item in payload.topics]
        prompt = (
            "You are a study planning copilot. Convert user instruction into topic setting updates.\n"
            "Return JSON object with keys:\n"
            "reply: short assistant message\n"
            "updates: array of objects with fields title, priority, target_minutes, difficulty, "
            "has_deadline, deadline.\n"
            "Rules:\n"
            "- Only update existing titles from provided topics.\n"
            "- Keep priority between 0.1 and 100.\n"
            "- Keep difficulty between 0.5 and 3.0.\n"
            "- Keep target_minutes between 0 and 5000.\n"
            "- deadline should be YYYY-MM-DD when has_deadline true.\n"
            "- Do not invent new topics.\n"
            f"Topics: {json.dumps(topic_rows)}\n"
            f"User message: {payload.message}"
        )
        raw = llm.invoke(prompt)
        content = str(getattr(raw, "content", raw))
        parsed = json.loads(content)
        updates: list[ChatUpdate] = []
        for candidate in parsed.get("updates", []):
            try:
                updates.append(ChatUpdate.model_validate(candidate))
            except Exception:
                continue
        return {
            "reply": parsed.get("reply", "Applied your requested planning adjustments."),
            "updates": [update.model_dump(mode="json") for update in updates],
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Chat adjustment failed: {exc}") from exc
