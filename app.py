"""End-to-end entry point for the Personalization Engine."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from core.calendar_sync import CalendarSyncConfig, CalendarSyncError, sync_study_blocks_to_calendar
from core.llm_provider import get_llm_provider_config
from core.optimizer import DailyLimits, DeepWorkWindow, StudyBlock, StudyScheduler
from core.parser import parse_syllabus_pdf
from core.rag import SyllabusRAG
from models.syllabus import ParsedSyllabus, Topic

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_window(window_text: str) -> DeepWorkWindow:
    """
    Parse availability like `mon 19:00-21:00`.

    Weekdays: mon,tue,wed,thu,fri,sat,sun.
    """
    days = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    try:
        day_str, times = window_text.strip().lower().split()
        start_str, end_str = times.split("-")
        weekday = days[day_str]
        start_t = datetime.strptime(start_str, "%H:%M").time()
        end_t = datetime.strptime(end_str, "%H:%M").time()
        return DeepWorkWindow(weekday=weekday, start_time=start_t, end_time=end_t)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Invalid window '{window_text}'. Expected format: 'mon 19:00-21:00'"
        ) from exc


def parse_syllabus(path: str, use_llm: bool) -> ParsedSyllabus:
    if not Path(path).exists():
        raise FileNotFoundError(f"Syllabus PDF does not exist: {path}")
    return parse_syllabus_pdf(path, use_llm=use_llm)


def ensure_llm_api_key(required: bool, context: str) -> None:
    """Raise a clear error when LLM features are requested without credentials."""
    if not required:
        return
    try:
        cfg = get_llm_provider_config()
    except ValueError as exc:
        raise ValueError(
            f"{context} requires API credentials. Set OPENAI_API_KEY (OpenAI) "
            f"or OPENROUTER_API_KEY (OpenRouter). Details: {exc}"
        ) from exc
    logger.info("Using LLM provider: %s", cfg.provider)


def optional_rag_query(syllabus: ParsedSyllabus, question: Optional[str]) -> Optional[str]:
    if not question:
        return None
    rag = SyllabusRAG()
    rag.ingest_text(syllabus.raw_text or "")
    return rag.query(question)


def _fallback_topics_from_raw_text(raw_text: str, max_topics: int = 8) -> list[Topic]:
    """Extract strict syllabus topic headings from raw text when LLM topics are missing."""
    if not raw_text.strip():
        return []
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    candidates: list[str] = []

    # Strict: capture explicit syllabus structure first (Unit/Module/Week headings),
    # including markdown bold wrappers seen in extracted PDF text.
    strict_patterns = [
        re.compile(r"^\*{0,2}\s*unit\s*\d+\s*:\s*.+?\*{0,2}$", re.IGNORECASE),
        re.compile(r"^\*{0,2}\s*module\s*\d+\s*:\s*.+?\*{0,2}$", re.IGNORECASE),
        re.compile(r"^\*{0,2}\s*chapter\s*\d+\s*:\s*.+?\*{0,2}$", re.IGNORECASE),
        re.compile(r"^\*{0,2}\s*weeks?\s*\d+(\s*[–-]\s*\d+)?\s*:\s*.+?\*{0,2}$", re.IGNORECASE),
    ]

    def _clean_heading(line: str) -> str:
        line = re.sub(r"^\*+|\*+$", "", line).strip()
        line = re.sub(r"\s+\(\d+%?\)\s*$", "", line).strip()
        return line

    for line in lines:
        if any(pattern.match(line) for pattern in strict_patterns):
            candidates.append(_clean_heading(line)[:140])
        if len(candidates) >= max_topics:
            break

    # Secondary strict mode: numbered syllabus headings like "1) Intro to AI".
    if not candidates:
        numbered_heading = re.compile(
            r"^\*{0,2}\s*[0-9]+[\.\)]\s+[A-Za-z].+?\*{0,2}$",
            re.IGNORECASE,
        )
        for line in lines:
            if numbered_heading.match(line):
                candidates.append(_clean_heading(line)[:140])
            if len(candidates) >= max_topics:
                break

    # Last-resort fallback only if strict extraction fails completely.
    if not candidates:
        generic_heading_pattern = re.compile(
            r"^(week\s*\d+|unit\s*\d+|module\s*\d+|chapter\s*\d+|[0-9]+[\.\)]\s+.+)$",
            re.IGNORECASE,
        )
        for line in lines:
            if generic_heading_pattern.match(line):
                candidates.append(_clean_heading(line)[:140])
            if len(candidates) >= max_topics:
                break

    if not candidates:
        # Coarse chunk fallback if headings are not obvious in extracted PDF text.
        chunk_size = max(1, len(lines) // max_topics)
        for i in range(0, len(lines), chunk_size):
            snippet = " ".join(lines[i : i + min(chunk_size, 3)])[:120]
            if snippet:
                candidates.append(f"Topic {len(candidates) + 1}: {snippet}")
            if len(candidates) >= max_topics:
                break

    cleaned: list[str] = []
    for title in candidates:
        t = _clean_heading(title)
        if t:
            cleaned.append(t)
    return [Topic(title=title, weightage_percent=None) for title in cleaned[:max_topics]]


def _dedupe_topic_titles(topics: list[Topic]) -> list[Topic]:
    """Deduplicate topics while preserving order."""
    seen: set[str] = set()
    unique: list[Topic] = []
    for topic in topics:
        key = topic.title.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(topic)
    return unique


def ensure_topics_for_scheduling(syllabus: ParsedSyllabus) -> ParsedSyllabus:
    """Guarantee the optimizer receives robust topic coverage."""
    # If LLM extracted very few topics, enrich with deterministic heading extraction.
    min_expected_topics = 5
    if len(syllabus.topics) >= min_expected_topics:
        syllabus.topics = _dedupe_topic_titles(syllabus.topics)
        return syllabus

    recovered = _fallback_topics_from_raw_text(syllabus.raw_text)
    if recovered:
        original_count = len(_dedupe_topic_titles(syllabus.topics))
        merged = _dedupe_topic_titles([*syllabus.topics, *recovered])
        syllabus.topics = merged
        logger.info(
            "Topic enrichment applied: %s total topics (%s fallback added).",
            len(merged),
            max(0, len(merged) - original_count),
        )
    return syllabus


def _inject_streamlit_style() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.1rem; padding-bottom: 1.4rem;}
        .app-card {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
        }
        .app-muted { color: #9CA3AF; font-size: 0.9rem; }
        .app-label { color: #E5E7EB; font-weight: 600; margin-bottom: 0.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _compute_parse_quality(raw_text: str, topics_count: int) -> dict[str, float | int]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    words = re.findall(r"[A-Za-z]{3,}", raw_text)
    heading_like = [
        line
        for line in lines
        if re.match(
            r"^(unit|module|chapter|week|topic|[0-9]+[\.\)])",
            line,
            flags=re.IGNORECASE,
        )
    ]
    suspicious_short_lines = [line for line in lines if len(line) <= 2]
    repeated_words = Counter([w.lower() for w in words])
    top_repetition = repeated_words.most_common(1)[0][1] if repeated_words else 0
    quality_score = 100
    if len(words) < 120:
        quality_score -= 25
    if not heading_like:
        quality_score -= 20
    if topics_count < 5:
        quality_score -= 20
    if top_repetition > max(30, len(words) * 0.08):
        quality_score -= 10
    if suspicious_short_lines and len(suspicious_short_lines) / max(1, len(lines)) > 0.08:
        quality_score -= 10

    return {
        "score": max(5, quality_score),
        "lines": len(lines),
        "words": len(words),
        "heading_like_lines": len(heading_like),
    }


def _build_schedule_insights(
    syllabus: ParsedSyllabus,
    topic_minutes_override: dict[str, int],
    topic_difficulty: dict[str, float],
    topic_deadlines: dict[str, date],
    blocks: list[StudyBlock],
) -> list[dict[str, object]]:
    planned_by_topic: dict[str, int] = {}
    overdue_by_topic: dict[str, int] = {}
    for block in blocks:
        planned_by_topic[block.topic] = planned_by_topic.get(block.topic, 0) + block.duration_minutes
        deadline = topic_deadlines.get(block.topic)
        if deadline and block.date > deadline:
            overdue_by_topic[block.topic] = overdue_by_topic.get(block.topic, 0) + block.duration_minutes

    analysis_rows: list[dict[str, object]] = []
    for topic in syllabus.topics:
        target_minutes = int(topic_minutes_override.get(topic.title, 0))
        planned_minutes = int(planned_by_topic.get(topic.title, 0))
        priority = float(topic.weightage_percent or 1.0)
        difficulty = float(topic_difficulty.get(topic.title, 1.0))
        deadline = topic_deadlines.get(topic.title)
        days_left = (deadline - date.today()).days if deadline else None
        deadline_urgency = 1.0
        if deadline is not None:
            if days_left <= 0:
                deadline_urgency = 2.5
            else:
                deadline_urgency = min(2.5, 1.0 + max(0.0, (30 - days_left) / 20.0))
        strict_score = round(priority * difficulty * deadline_urgency, 3)
        coverage = round((planned_minutes / target_minutes) * 100, 1) if target_minutes > 0 else None
        analysis_rows.append(
            {
                "topic": topic.title,
                "priority": round(priority, 2),
                "difficulty": round(difficulty, 2),
                "target_minutes": target_minutes,
                "planned_minutes": planned_minutes,
                "coverage_pct": coverage,
                "deadline": deadline.isoformat() if deadline else None,
                "deadline_urgency": round(deadline_urgency, 2),
                "strict_score": strict_score,
                "overdue_minutes": int(overdue_by_topic.get(topic.title, 0)),
            }
        )
    return analysis_rows


def _render_priority_editor(syllabus: ParsedSyllabus) -> tuple[dict[str, int], dict[str, float], dict[str, date]]:
    import streamlit as st

    st.subheader("Topic Priorities and Time Allocation")
    st.caption(
        "Edit priority, target minutes, difficulty, and optional deadline in one table."
    )
    today_plus_14 = date.today() + timedelta(days=14)
    rows: list[dict[str, object]] = []
    for topic in syllabus.topics:
        rows.append(
            {
                "topic": topic.title,
                "priority": float(topic.weightage_percent or 1.0),
                "target_minutes": int((topic.estimated_hours or 0) * 60),
                "difficulty": 1.0,
                "has_deadline": False,
                "deadline": today_plus_14,
            }
        )

    edited = st.data_editor(
        rows,
        use_container_width=True,
        hide_index=True,
        key="topic_priority_editor",
        column_config={
            "topic": st.column_config.TextColumn("Topic", disabled=True, width="large"),
            "priority": st.column_config.NumberColumn("Priority", min_value=0.1, max_value=100.0, step=0.1),
            "target_minutes": st.column_config.NumberColumn("Target Minutes", min_value=0, max_value=5000, step=15),
            "difficulty": st.column_config.NumberColumn("Difficulty", min_value=0.5, max_value=3.0, step=0.1),
            "has_deadline": st.column_config.CheckboxColumn("Has Deadline"),
            "deadline": st.column_config.DateColumn("Due Date", format="YYYY-MM-DD"),
        },
    )

    records = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
    topic_minutes_override: dict[str, int] = {}
    topic_difficulty: dict[str, float] = {}
    topic_deadlines: dict[str, date] = {}

    for item in records:
        title = str(item["topic"])
        minutes = int(item.get("target_minutes", 0) or 0)
        priority = float(item.get("priority", 1.0) or 1.0)
        difficulty = float(item.get("difficulty", 1.0) or 1.0)
        has_deadline = bool(item.get("has_deadline", False))
        deadline_value = item.get("deadline")

        for topic in syllabus.topics:
            if topic.title == title:
                topic.weightage_percent = priority
                topic.estimated_hours = float(minutes) / 60 if minutes > 0 else None
                break

        topic_minutes_override[title] = minutes
        topic_difficulty[title] = difficulty

        if has_deadline and deadline_value:
            if isinstance(deadline_value, datetime):
                topic_deadlines[title] = deadline_value.date()
            elif isinstance(deadline_value, date):
                topic_deadlines[title] = deadline_value
            elif isinstance(deadline_value, str):
                try:
                    topic_deadlines[title] = date.fromisoformat(deadline_value)
                except ValueError:
                    pass

    return topic_minutes_override, topic_difficulty, topic_deadlines


def _render_schedule_results(blocks: list[StudyBlock], analysis_rows: list[dict[str, object]]) -> None:
    import streamlit as st

    st.subheader("Generated Study Plan")
    schedule_tab, agenda_tab, analysis_tab = st.tabs(
        ["Schedule Table", "Daily Agenda", "Priority Analysis"]
    )

    with schedule_tab:
        st.dataframe([asdict(b) for b in blocks], use_container_width=True)

    with agenda_tab:
        by_date: dict[date, list[StudyBlock]] = {}
        for block in sorted(blocks, key=lambda b: (b.date, b.start_time)):
            by_date.setdefault(block.date, []).append(block)
        for block_date, day_blocks in by_date.items():
            total = sum(b.duration_minutes for b in day_blocks)
            with st.expander(f"{block_date.isoformat()}  -  {total} minutes", expanded=False):
                for b in day_blocks:
                    st.write(
                        f"- {b.start_time.strftime('%H:%M')} ({b.duration_minutes}m) [{b.type}] {b.topic}"
                    )

    with analysis_tab:
        st.dataframe(analysis_rows, use_container_width=True)
        under_planned = [r for r in analysis_rows if r["coverage_pct"] is not None and r["coverage_pct"] < 80]
        overdue_topics = [r for r in analysis_rows if int(r["overdue_minutes"]) > 0]
        flagged_col1, flagged_col2 = st.columns(2)
        with flagged_col1:
            st.metric("Under-planned topics (<80% target)", len(under_planned))
        with flagged_col2:
            st.metric("Topics with overdue minutes", len(overdue_topics))

        if under_planned:
            st.warning(
                "Some topics are under-planned against target minutes. Increase priority/difficulty "
                "or relax no-study constraints."
            )
        if overdue_topics:
            st.error(
                "Some planned minutes are after topic deadlines. Tighten deadlines planning or add more windows."
            )


def print_schedule(blocks: list[StudyBlock]) -> None:
    if not blocks:
        print("No blocks were generated.")
        return
    print("\nGenerated Schedule:")
    for block in blocks:
        print(
            f"- {block.date.isoformat()} {block.start_time.strftime('%H:%M')} "
            f"({block.duration_minutes}m) [{block.type}] {block.topic}"
        )


def run_pipeline(args: argparse.Namespace) -> None:
    load_dotenv()
    setup_logging(args.log_level)
    logger.info("Starting personalization pipeline.")

    if not args.pdf:
        raise ValueError("Please pass --pdf for CLI mode, or run with Streamlit UI.")

    ensure_llm_api_key(args.use_llm, "LLM syllabus parsing")
    ensure_llm_api_key(bool(args.query), "RAG querying")

    syllabus = parse_syllabus(args.pdf, use_llm=args.use_llm)
    syllabus = ensure_topics_for_scheduling(syllabus)
    logger.info("Parsed syllabus with %s topics and %s exam dates.", len(syllabus.topics), len(syllabus.exam_dates))

    answer = optional_rag_query(syllabus, args.query)
    if answer:
        print("\nRAG Answer:")
        print(answer)

    default_windows = ["mon 19:00-21:00", "wed 19:00-21:00", "sat 10:00-12:00"]
    window_inputs = args.window if args.window else default_windows
    windows = [parse_window(w) for w in window_inputs]
    limits = DailyLimits(
        max_minutes_per_day=args.max_minutes_per_day,
        min_block_minutes=args.min_block_minutes,
        max_block_minutes=args.max_block_minutes,
    )

    scheduler = StudyScheduler(preferred_mode=args.optimizer_mode)
    blocks = scheduler.build_schedule(
        syllabus=syllabus,
        deep_work_windows=windows,
        daily_limits=limits,
        include_reviews=not args.disable_reviews,
    )
    print_schedule(blocks)

    if args.sync_calendar:
        config = CalendarSyncConfig(
            credentials_path=args.google_credentials,
            token_path=args.google_token,
            calendar_id=args.calendar_id,
            timezone=args.timezone,
            reminder_minutes_before=args.reminder_minutes,
        )
        try:
            ids = sync_study_blocks_to_calendar(blocks, config=config)
            print(f"\nSynced {len(ids)} events to Google Calendar ({config.calendar_id}).")
        except CalendarSyncError as exc:
            logger.error("Calendar sync failed: %s", exc)
            print(f"\nCalendar sync failed: {exc}")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="24/7 Personalized Teaching Assistant")
    parser.add_argument("--pdf", default=None, help="Path to syllabus PDF file.")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM parsing.")
    parser.add_argument("--query", default=None, help="Optional RAG question.")
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        help="Deep-work window, e.g., --window 'mon 19:00-21:00'. Repeat flag for multiple.",
    )
    parser.add_argument("--optimizer-mode", choices=["cp_sat", "greedy"], default="cp_sat")
    parser.add_argument("--max-minutes-per-day", type=int, default=180)
    parser.add_argument("--min-block-minutes", type=int, default=30)
    parser.add_argument("--max-block-minutes", type=int, default=120)
    parser.add_argument("--disable-reviews", action="store_true")

    parser.add_argument("--sync-calendar", action="store_true")
    parser.add_argument("--google-credentials", default="credentials.json")
    parser.add_argument("--google-token", default="token.json")
    parser.add_argument("--calendar-id", default="primary")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--reminder-minutes", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    return parser


def run_streamlit_demo() -> None:
    """
    Lightweight Streamlit demo.

    Run with:
        streamlit run app.py
    """
    import streamlit as st

    load_dotenv()
    _inject_streamlit_style()
    st.title("24/7 Personalized Teaching Assistant")
    st.write("Upload your syllabus, verify parsing quality, and generate a focused study plan.")

    st.markdown('<div class="app-card"><div class="app-label">Workflow</div><div class="app-muted">1) Upload PDF  2) Parse + validate  3) Tune priorities/deadlines  4) Generate and review schedule analysis</div></div>', unsafe_allow_html=True)

    col_upload, col_options = st.columns([1.3, 1.1])
    with col_upload:
        uploaded = st.file_uploader("Upload syllabus PDF", type=["pdf"])
        query = st.text_input("Optional RAG question")
    with col_options:
        use_llm = st.checkbox("Use LLM parsing", value=True)
        optimizer_mode = st.selectbox("Optimizer Mode", options=["cp_sat", "greedy"], index=0)
        include_reviews = st.checkbox("Include spaced-repetition reviews", value=True)
        strict_mode = st.checkbox(
            "Strict analyzed scheduling mode",
            value=True,
            help="Prioritizes allocation using priority, remaining duration, and deadline urgency.",
        )
    weekday_to_idx = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    no_study_labels = st.multiselect(
        "No-study weekdays",
        options=list(weekday_to_idx.keys()),
        default=[],
        help="Selected weekdays are blocked from scheduling.",
    )

    if uploaded is None:
        return

    upload_signature = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("uploaded_signature") != upload_signature:
        st.session_state["uploaded_signature"] = upload_signature
        st.session_state.pop("parsed_syllabus_json", None)

    if st.button("Parse Syllabus", type="primary"):
        temp_path = Path(".streamlit_temp_syllabus.pdf")
        temp_path.write_bytes(uploaded.getvalue())
        try:
            ensure_llm_api_key(use_llm, "LLM syllabus parsing")
            parsed = parse_syllabus_pdf(str(temp_path), use_llm=use_llm)
            parsed = ensure_topics_for_scheduling(parsed)
            if not parsed.topics:
                st.error(
                    "Could not extract usable topics from this PDF. Try enabling LLM parsing "
                    "or upload a text-selectable syllabus PDF."
                )
                return
            st.session_state["parsed_syllabus_json"] = parsed.model_dump_json()
            st.success(f"Syllabus parsed successfully. Topics detected: {len(parsed.topics)}")
            quality = _compute_parse_quality(parsed.raw_text or "", len(parsed.topics))
            st.session_state["parse_quality"] = quality
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
        finally:
            if temp_path.exists():
                temp_path.unlink()

    if "parsed_syllabus_json" not in st.session_state:
        st.info("Upload your PDF and click 'Parse Syllabus' to configure priorities/deadlines.")
        return

    syllabus = ParsedSyllabus.model_validate_json(st.session_state["parsed_syllabus_json"])
    parse_quality = st.session_state.get("parse_quality", _compute_parse_quality(syllabus.raw_text or "", len(syllabus.topics)))

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Topics", len(syllabus.topics))
    metric_col2.metric("Raw text lines", int(parse_quality["lines"]))
    metric_col3.metric("Heading-like lines", int(parse_quality["heading_like_lines"]))
    metric_col4.metric("Parse quality", f'{int(parse_quality["score"])} / 100')

    if int(parse_quality["score"]) < 60:
        st.warning(
            "Parsing quality looks low. Try enabling LLM parsing, using a text-selectable PDF, "
            "or re-uploading a clearer file."
        )

    with st.expander("Preview extracted syllabus text"):
        preview = (syllabus.raw_text or "").strip()
        st.text_area(
            "Extracted text preview",
            value=preview[:2500] if preview else "No text extracted.",
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    topic_minutes_override, topic_difficulty, topic_deadlines = _render_priority_editor(syllabus)

    if not st.button("Generate Schedule", type="primary"):
        return

    try:
        ensure_llm_api_key(bool(query), "RAG querying")
        if query:
            ans = optional_rag_query(syllabus, query)
            st.subheader("RAG Answer")
            st.write(ans)

        windows = [
            DeepWorkWindow(weekday=0, start_time=time(19, 0), end_time=time(21, 0)),
            DeepWorkWindow(weekday=2, start_time=time(19, 0), end_time=time(21, 0)),
            DeepWorkWindow(weekday=5, start_time=time(10, 0), end_time=time(12, 0)),
        ]
        explicit_minutes_mode = any(v > 0 for v in topic_minutes_override.values())
        scheduler = StudyScheduler(preferred_mode=optimizer_mode)
        blocks = scheduler.build_schedule(
            syllabus=syllabus,
            deep_work_windows=windows,
            daily_limits=DailyLimits(),
            include_reviews=include_reviews,
            topic_minutes_override=topic_minutes_override if explicit_minutes_mode else None,
            topic_difficulty=topic_difficulty,
            topic_deadlines=topic_deadlines,
            no_study_weekdays={weekday_to_idx[d] for d in no_study_labels},
            strict_mode=strict_mode,
        )
        if blocks:
            analysis_rows = _build_schedule_insights(
                syllabus=syllabus,
                topic_minutes_override=topic_minutes_override,
                topic_difficulty=topic_difficulty,
                topic_deadlines=topic_deadlines,
                blocks=blocks,
            )
            _render_schedule_results(blocks, analysis_rows)
        else:
            st.info("No schedule generated.")
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))


def _is_running_with_streamlit() -> bool:
    """Return True when this script is executed by `streamlit run`."""
    if "streamlit" in sys.argv[0].lower():
        return True
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    if _is_running_with_streamlit():
        run_streamlit_demo()
    else:
        cli = build_cli()
        cli_args = cli.parse_args()
        run_pipeline(cli_args)
