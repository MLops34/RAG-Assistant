"""Pydantic models for syllabus parsing and downstream scheduling/RAG."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Topic(BaseModel):
    title: str
    description: Optional[str] = None
    weightage_percent: Optional[float] = Field(
        default=None, description="Approximate grade weight for this unit/topic if stated."
    )
    learning_objectives: List[str] = Field(default_factory=list)
    week_or_unit: Optional[str] = None


class ExamEvent(BaseModel):
    name: str
    date: Optional[str] = Field(
        default=None, description="ISO date YYYY-MM-DD if inferable from the syllabus."
    )
    weightage_percent: Optional[float] = None


class ParsedSyllabus(BaseModel):
    course_title: Optional[str] = None
    instructor: Optional[str] = None
    topics: List[Topic] = Field(default_factory=list)
    exams: List[ExamEvent] = Field(default_factory=list)
    raw_text: str = ""
    source_path: Optional[str] = None
