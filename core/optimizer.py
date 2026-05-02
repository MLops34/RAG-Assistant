"""Study plan optimization using CP-SAT with greedy fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal, Optional

from models.syllabus import ParsedSyllabus

try:
    from ortools.sat.python import cp_model  # type: ignore[reportMissingImports]

    _ORTOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    cp_model = None
    _ORTOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)

BlockType = Literal["study", "review"]


@dataclass(frozen=True)
class DeepWorkWindow:
    """A recurring weekly time slot available for focused study."""

    weekday: int  # 0=Monday ... 6=Sunday
    start_time: time
    end_time: time


@dataclass(frozen=True)
class DailyLimits:
    """Daily study constraints used by both scheduling strategies."""

    max_minutes_per_day: int = 180
    min_block_minutes: int = 30
    max_block_minutes: int = 120

    def validate(self) -> None:
        if self.max_minutes_per_day <= 0:
            raise ValueError("max_minutes_per_day must be positive.")
        if self.min_block_minutes <= 0:
            raise ValueError("min_block_minutes must be positive.")
        if self.max_block_minutes < self.min_block_minutes:
            raise ValueError("max_block_minutes must be >= min_block_minutes.")


@dataclass(frozen=True)
class StudyBlock:
    """A concrete scheduled study/review event."""

    topic: str
    date: date
    start_time: time
    duration_minutes: int
    type: BlockType

    @property
    def end_time(self) -> time:
        start_dt = datetime.combine(self.date, self.start_time)
        return (start_dt + timedelta(minutes=self.duration_minutes)).time()


class StudyScheduler:
    """
    Build a study plan from parsed syllabus data and availability constraints.

    Preferred mode is CP-SAT (global optimization). If OR-Tools is unavailable
    or solving fails, greedy scheduling is used as a safe fallback.
    """

    DEFAULT_REVIEW_OFFSETS_DAYS = (1, 3, 7)

    def __init__(self, *, preferred_mode: Literal["cp_sat", "greedy"] = "cp_sat") -> None:
        self.preferred_mode = preferred_mode

    def build_schedule(
        self,
        syllabus: ParsedSyllabus,
        deep_work_windows: list[DeepWorkWindow],
        daily_limits: DailyLimits,
        *,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        include_reviews: bool = True,
        topic_minutes_override: Optional[dict[str, int]] = None,
        topic_difficulty: Optional[dict[str, float]] = None,
        topic_deadlines: Optional[dict[str, date]] = None,
        no_study_days: Optional[set[date]] = None,
        no_study_weekdays: Optional[set[int]] = None,
        strict_mode: bool = False,
    ) -> list[StudyBlock]:
        """
        Generate study blocks from syllabus topics and user constraints.

        Args:
            syllabus: Parsed syllabus with topics and exam milestones.
            deep_work_windows: Weekly windows available for deep work.
            daily_limits: Daily and per-block duration constraints.
            start_date: Optional planning start (default: today).
            end_date: Optional planning horizon. If absent, inferred from nearest exam.
            include_reviews: Whether spaced-repetition review blocks should be added.
            topic_minutes_override: Optional explicit per-topic minutes budget.
            topic_difficulty: Optional per-topic difficulty multiplier map.
            topic_deadlines: Optional per-topic deadlines to increase urgency.
            no_study_days: Optional specific dates to avoid scheduling.
            no_study_weekdays: Optional weekdays to avoid (0=Mon .. 6=Sun).
            strict_mode: If True, enforce deadline-aware greedy ranking.
        """
        daily_limits.validate()
        if not deep_work_windows:
            raise ValueError("deep_work_windows cannot be empty.")
        if not syllabus.topics:
            raise ValueError("No topics found in syllabus. Cannot build schedule.")

        planning_start = start_date or date.today()
        planning_end = end_date or self._infer_horizon_end(syllabus, planning_start)
        if planning_end < planning_start:
            raise ValueError("end_date must be on or after start_date.")

        windows = self._materialize_windows(
            deep_work_windows,
            planning_start,
            planning_end,
            no_study_days=no_study_days,
            no_study_weekdays=no_study_weekdays,
        )
        if not windows:
            raise ValueError("No usable deep-work windows within planning horizon.")

        if topic_minutes_override:
            topic_minutes = self._validated_topic_minutes_override(
                topics=[topic.title for topic in syllabus.topics],
                overrides=topic_minutes_override,
                daily_limits=daily_limits,
            )
        else:
            weights = self._topic_weights(
                syllabus,
                planning_start=planning_start,
                topic_difficulty=topic_difficulty,
                topic_deadlines=topic_deadlines,
            )
            topic_minutes = self._topic_target_minutes(
                topic_weights=weights,
                windows=windows,
                daily_limits=daily_limits,
            )

        mode = self.preferred_mode
        if strict_mode and mode == "cp_sat":
            logger.info("Strict mode enabled: using greedy scheduler for deadline-aware ranking.")
            mode = "greedy"
        if mode == "cp_sat" and not _ORTOOLS_AVAILABLE:
            logger.warning("OR-Tools is not installed; falling back to greedy scheduler.")
            mode = "greedy"

        blocks: list[StudyBlock]
        if mode == "cp_sat":
            blocks = self._schedule_with_cp_sat(
                topic_minutes=topic_minutes,
                windows=windows,
                daily_limits=daily_limits,
            )
            if not blocks:
                logger.warning("CP-SAT did not produce a feasible plan; using greedy fallback.")
                blocks = self._schedule_with_greedy(
                    topic_minutes=topic_minutes,
                    windows=windows,
                    daily_limits=daily_limits,
                    topic_priority_weights=weights,
                    topic_deadlines=topic_deadlines,
                    strict_mode=strict_mode,
                )
        else:
            blocks = self._schedule_with_greedy(
                topic_minutes=topic_minutes,
                windows=windows,
                daily_limits=daily_limits,
                topic_priority_weights=weights if "weights" in locals() else None,
                topic_deadlines=topic_deadlines,
                strict_mode=strict_mode,
            )

        if include_reviews:
            blocks = self._inject_spaced_repetition_reviews(
                study_blocks=blocks,
                windows=windows,
                daily_limits=daily_limits,
            )

        return sorted(blocks, key=lambda b: (b.date, b.start_time))

    def _infer_horizon_end(self, syllabus: ParsedSyllabus, planning_start: date) -> date:
        exam_dates = [exam.date for exam in syllabus.exam_dates if exam.date is not None]
        if exam_dates:
            nearest = min(d for d in exam_dates if d >= planning_start) if any(
                d >= planning_start for d in exam_dates
            ) else max(exam_dates)
            return nearest
        return planning_start + timedelta(days=28)

    def _topic_weights(
        self,
        syllabus: ParsedSyllabus,
        *,
        planning_start: date,
        topic_difficulty: Optional[dict[str, float]],
        topic_deadlines: Optional[dict[str, date]],
    ) -> dict[str, float]:
        weights: dict[str, float] = {}
        fallback = 1.0
        for topic in syllabus.topics:
            base = topic.weightage_percent if topic.weightage_percent is not None else fallback
            difficulty = max(0.2, float((topic_difficulty or {}).get(topic.title, 1.0)))
            deadline_factor = self._deadline_urgency_multiplier(
                deadline=(topic_deadlines or {}).get(topic.title),
                planning_start=planning_start,
            )
            value = base * difficulty * deadline_factor
            weights[topic.title] = max(value, 0.1)
        return weights

    def _deadline_urgency_multiplier(self, *, deadline: Optional[date], planning_start: date) -> float:
        if deadline is None:
            return 1.0
        days_left = (deadline - planning_start).days
        if days_left <= 0:
            return 2.5
        # Boost urgency as deadline approaches (up to 2.5x).
        return min(2.5, 1.0 + max(0.0, (30 - days_left) / 20.0))

    def _materialize_windows(
        self,
        windows: list[DeepWorkWindow],
        start_date: date,
        end_date: date,
        *,
        no_study_days: Optional[set[date]] = None,
        no_study_weekdays: Optional[set[int]] = None,
    ) -> list[dict[str, object]]:
        materialized: list[dict[str, object]] = []
        blocked_days = no_study_days or set()
        blocked_weekdays = no_study_weekdays or set()
        day = start_date
        while day <= end_date:
            if day in blocked_days or day.weekday() in blocked_weekdays:
                day += timedelta(days=1)
                continue
            for w in windows:
                if day.weekday() != w.weekday:
                    continue
                start_dt = datetime.combine(day, w.start_time)
                end_dt = datetime.combine(day, w.end_time)
                if end_dt <= start_dt:
                    logger.warning("Skipping invalid window on weekday=%s (%s <= %s).", w.weekday, w.end_time, w.start_time)
                    continue
                materialized.append(
                    {
                        "date": day,
                        "start": start_dt,
                        "end": end_dt,
                        "capacity": int((end_dt - start_dt).total_seconds() // 60),
                    }
                )
            day += timedelta(days=1)
        return materialized

    def _topic_target_minutes(
        self,
        topic_weights: dict[str, float],
        windows: list[dict[str, object]],
        daily_limits: DailyLimits,
    ) -> dict[str, int]:
        total_capacity = 0
        by_day: dict[date, int] = {}
        for slot in windows:
            day = slot["date"]
            cap = min(int(slot["capacity"]), daily_limits.max_minutes_per_day)
            by_day[day] = by_day.get(day, 0) + cap
        total_capacity = sum(min(v, daily_limits.max_minutes_per_day) for v in by_day.values())

        weight_sum = sum(topic_weights.values()) or 1.0
        minutes: dict[str, int] = {}
        for topic, weight in topic_weights.items():
            raw = int(total_capacity * (weight / weight_sum))
            # Force at least one minimum block per topic.
            minutes[topic] = max(daily_limits.min_block_minutes, raw)
        return minutes

    def _validated_topic_minutes_override(
        self,
        *,
        topics: list[str],
        overrides: dict[str, int],
        daily_limits: DailyLimits,
    ) -> dict[str, int]:
        resolved: dict[str, int] = {}
        for title in topics:
            requested = int(overrides.get(title, 0))
            if requested <= 0:
                continue
            resolved[title] = max(daily_limits.min_block_minutes, requested)
        if not resolved:
            raise ValueError(
                "topic_minutes_override was provided but no topic had positive minutes."
            )
        return resolved

    def _schedule_with_cp_sat(
        self,
        *,
        topic_minutes: dict[str, int],
        windows: list[dict[str, object]],
        daily_limits: DailyLimits,
    ) -> list[StudyBlock]:
        if not _ORTOOLS_AVAILABLE or cp_model is None:
            return []

        model = cp_model.CpModel()
        topic_names = list(topic_minutes.keys())

        slot_minutes = min(daily_limits.min_block_minutes, 30)
        units_per_topic = {
            t: max(1, (m + slot_minutes - 1) // slot_minutes) for t, m in topic_minutes.items()
        }

        assignment: dict[tuple[int, int, int], cp_model.IntVar] = {}
        slot_offsets: list[list[datetime]] = []
        for i, window in enumerate(windows):
            window_start = window["start"]
            capacity = int(window["capacity"])
            slot_count = capacity // slot_minutes
            offsets: list[datetime] = []
            for s in range(slot_count):
                offsets.append(window_start + timedelta(minutes=s * slot_minutes))
            slot_offsets.append(offsets)

            for s in range(slot_count):
                for t_idx, _ in enumerate(topic_names):
                    assignment[(i, s, t_idx)] = model.NewBoolVar(f"a_w{i}_s{s}_t{t_idx}")

                model.Add(
                    sum(assignment[(i, s, t_idx)] for t_idx in range(len(topic_names))) <= 1
                )

        for t_idx, topic in enumerate(topic_names):
            model.Add(
                sum(
                    assignment[(i, s, t_idx)]
                    for i, offsets in enumerate(slot_offsets)
                    for s in range(len(offsets))
                )
                == units_per_topic[topic]
            )

        day_to_slots: dict[date, list[tuple[int, int, int]]] = {}
        for i, window in enumerate(windows):
            day = window["date"]
            for s in range(len(slot_offsets[i])):
                for t_idx in range(len(topic_names)):
                    day_to_slots.setdefault(day, []).append((i, s, t_idx))

        max_slots_per_day = daily_limits.max_minutes_per_day // slot_minutes
        for _, vars_per_day in day_to_slots.items():
            model.Add(sum(assignment[key] for key in vars_per_day) <= max_slots_per_day)

        objective_terms = []
        for t_idx, topic in enumerate(topic_names):
            for i, offsets in enumerate(slot_offsets):
                for s in range(len(offsets)):
                    # Earlier slots are slightly preferred to avoid cramming.
                    recency_penalty = i * 2 + s
                    objective_terms.append((1000 - recency_penalty) * assignment[(i, s, t_idx)])
        model.Maximize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.warning("CP-SAT returned no feasible solution (status=%s).", status)
            return []

        blocks: list[StudyBlock] = []
        for i, offsets in enumerate(slot_offsets):
            for s, start_dt in enumerate(offsets):
                for t_idx, topic in enumerate(topic_names):
                    if solver.Value(assignment[(i, s, t_idx)]) != 1:
                        continue
                    blocks.append(
                        StudyBlock(
                            topic=topic,
                            date=start_dt.date(),
                            start_time=start_dt.time(),
                            duration_minutes=slot_minutes,
                            type="study",
                        )
                    )
        return self._merge_adjacent_blocks(blocks, daily_limits.max_block_minutes)

    def _schedule_with_greedy(
        self,
        *,
        topic_minutes: dict[str, int],
        windows: list[dict[str, object]],
        daily_limits: DailyLimits,
        topic_priority_weights: Optional[dict[str, float]] = None,
        topic_deadlines: Optional[dict[str, date]] = None,
        strict_mode: bool = False,
    ) -> list[StudyBlock]:
        remaining = dict(topic_minutes)
        ordered_topics = sorted(remaining, key=lambda t: remaining[t], reverse=True)
        per_day_used: dict[date, int] = {}
        blocks: list[StudyBlock] = []

        for window in sorted(windows, key=lambda w: w["start"]):
            day = window["date"]
            start_dt = window["start"]
            end_dt = window["end"]
            current = start_dt

            while current < end_dt and any(v > 0 for v in remaining.values()):
                if per_day_used.get(day, 0) >= daily_limits.max_minutes_per_day:
                    break

                ordered_topics.sort(
                    key=lambda t: self._strict_topic_score(
                        topic=t,
                        remaining=remaining,
                        current_day=day,
                        topic_priority_weights=topic_priority_weights,
                        topic_deadlines=topic_deadlines,
                        strict_mode=strict_mode,
                    ),
                    reverse=True,
                )
                topic = next(
                    (
                        t
                        for t in ordered_topics
                        if remaining[t] > 0
                        and not self._deadline_missed(
                            topic=t,
                            current_day=day,
                            topic_deadlines=topic_deadlines,
                            strict_mode=strict_mode,
                        )
                    ),
                    None,
                )
                if topic is None:
                    break

                available_window = int((end_dt - current).total_seconds() // 60)
                available_day = daily_limits.max_minutes_per_day - per_day_used.get(day, 0)
                if available_window < daily_limits.min_block_minutes:
                    break

                duration = min(
                    daily_limits.max_block_minutes,
                    available_window,
                    available_day,
                    remaining[topic],
                )
                duration = max(daily_limits.min_block_minutes, duration)
                duration = min(duration, available_window, available_day, remaining[topic])
                if duration < daily_limits.min_block_minutes:
                    break

                blocks.append(
                    StudyBlock(
                        topic=topic,
                        date=day,
                        start_time=current.time(),
                        duration_minutes=duration,
                        type="study",
                    )
                )
                remaining[topic] -= duration
                per_day_used[day] = per_day_used.get(day, 0) + duration
                current += timedelta(minutes=duration)

        return blocks

    def _deadline_missed(
        self,
        *,
        topic: str,
        current_day: date,
        topic_deadlines: Optional[dict[str, date]],
        strict_mode: bool,
    ) -> bool:
        if not strict_mode or not topic_deadlines:
            return False
        deadline = topic_deadlines.get(topic)
        return bool(deadline and current_day > deadline)

    def _strict_topic_score(
        self,
        *,
        topic: str,
        remaining: dict[str, int],
        current_day: date,
        topic_priority_weights: Optional[dict[str, float]],
        topic_deadlines: Optional[dict[str, date]],
        strict_mode: bool,
    ) -> float:
        if remaining.get(topic, 0) <= 0:
            return -1e9
        if self._deadline_missed(
            topic=topic,
            current_day=current_day,
            topic_deadlines=topic_deadlines,
            strict_mode=strict_mode,
        ):
            return -1e9
        weight = float((topic_priority_weights or {}).get(topic, 1.0))
        urgency = self._deadline_urgency_multiplier(
            deadline=(topic_deadlines or {}).get(topic),
            planning_start=current_day,
        )
        # Combined strict score: high priority + high remaining + urgent deadline.
        return weight * urgency * max(1.0, remaining[topic] / 30.0)

    def _inject_spaced_repetition_reviews(
        self,
        *,
        study_blocks: list[StudyBlock],
        windows: list[dict[str, object]],
        daily_limits: DailyLimits,
    ) -> list[StudyBlock]:
        if not study_blocks:
            return []

        daily_usage: dict[date, int] = {}
        for block in study_blocks:
            daily_usage[block.date] = daily_usage.get(block.date, 0) + block.duration_minutes

        windows_by_date: dict[date, list[dict[str, object]]] = {}
        for w in windows:
            windows_by_date.setdefault(w["date"], []).append(w)

        reviews: list[StudyBlock] = []
        for block in study_blocks:
            review_duration = max(15, block.duration_minutes // 3)
            for offset in self.DEFAULT_REVIEW_OFFSETS_DAYS:
                review_day = block.date + timedelta(days=offset)
                if review_day not in windows_by_date:
                    continue
                if daily_usage.get(review_day, 0) + review_duration > daily_limits.max_minutes_per_day:
                    continue
                slot = self._first_fittable_time(
                    day_windows=windows_by_date[review_day],
                    existing_blocks=study_blocks + reviews,
                    duration_minutes=review_duration,
                    target_day=review_day,
                )
                if slot is None:
                    continue

                reviews.append(
                    StudyBlock(
                        topic=block.topic,
                        date=review_day,
                        start_time=slot.time(),
                        duration_minutes=review_duration,
                        type="review",
                    )
                )
                daily_usage[review_day] = daily_usage.get(review_day, 0) + review_duration

        return study_blocks + reviews

    def _first_fittable_time(
        self,
        *,
        day_windows: list[dict[str, object]],
        existing_blocks: list[StudyBlock],
        duration_minutes: int,
        target_day: date,
    ) -> Optional[datetime]:
        day_windows = sorted(day_windows, key=lambda x: x["start"])
        occupied: list[tuple[datetime, datetime]] = []
        for block in existing_blocks:
            if block.date != target_day:
                continue
            start_dt = datetime.combine(block.date, block.start_time)
            end_dt = start_dt + timedelta(minutes=block.duration_minutes)
            occupied.append((start_dt, end_dt))
        occupied.sort(key=lambda pair: pair[0])

        for w in day_windows:
            cursor = w["start"]
            w_end = w["end"]
            for occ_start, occ_end in occupied:
                if occ_end <= cursor:
                    continue
                if occ_start > cursor and (occ_start - cursor).total_seconds() // 60 >= duration_minutes:
                    return cursor
                cursor = max(cursor, occ_end)
                if cursor >= w_end:
                    break
            if (w_end - cursor).total_seconds() // 60 >= duration_minutes:
                return cursor
        return None

    def _merge_adjacent_blocks(self, blocks: list[StudyBlock], max_block_minutes: int) -> list[StudyBlock]:
        if not blocks:
            return []
        blocks = sorted(blocks, key=lambda b: (b.date, b.start_time, b.topic))
        merged: list[StudyBlock] = [blocks[0]]

        for block in blocks[1:]:
            prev = merged[-1]
            prev_end = datetime.combine(prev.date, prev.start_time) + timedelta(minutes=prev.duration_minutes)
            curr_start = datetime.combine(block.date, block.start_time)
            if (
                prev.topic == block.topic
                and prev.type == block.type
                and prev.date == block.date
                and prev_end == curr_start
                and prev.duration_minutes + block.duration_minutes <= max_block_minutes
            ):
                merged[-1] = StudyBlock(
                    topic=prev.topic,
                    date=prev.date,
                    start_time=prev.start_time,
                    duration_minutes=prev.duration_minutes + block.duration_minutes,
                    type=prev.type,
                )
                continue
            merged.append(block)
        return merged
