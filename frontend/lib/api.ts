export type ParseQuality = {
  score: number;
  lines: number;
  words: number;
  heading_like_lines: number;
};

export type Topic = {
  title: string;
  priority: number;
  target_minutes: number;
  difficulty: number;
  has_deadline: boolean;
  deadline: string;
};

export type ParsedSyllabus = {
  topics: Array<{
    title: string;
    weightage_percent: number | null;
    estimated_hours: number | null;
  }>;
  raw_text: string;
  [key: string]: unknown;
};

export type ScheduleBlock = {
  topic: string;
  date: string;
  start_time: string;
  duration_minutes: number;
  type: string;
};

export type ScheduleAnalysis = {
  topic: string;
  priority: number;
  difficulty: number;
  target_minutes: number;
  planned_minutes: number;
  coverage_pct: number | null;
  deadline: string | null;
  deadline_urgency: number;
  strict_score: number;
  overdue_minutes: number;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatAdjustUpdate = {
  title: string;
  priority?: number;
  target_minutes?: number;
  difficulty?: number;
  has_deadline?: boolean;
  deadline?: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function parseSyllabus(file: File, useLlm: boolean) {
  const form = new FormData();
  form.append("file", file);
  form.append("use_llm", String(useLlm));
  const res = await fetch(`${API_BASE}/api/parse`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to parse syllabus");
  }
  return (await res.json()) as {
    syllabus: ParsedSyllabus;
    parse_quality: ParseQuality;
  };
}

export async function generateSchedule(input: {
  syllabus: ParsedSyllabus;
  topics: Topic[];
  optimizer_mode: "cp_sat" | "greedy";
  include_reviews: boolean;
  strict_mode: boolean;
  query?: string;
  no_study_weekdays: number[];
}) {
  const res = await fetch(`${API_BASE}/api/schedule`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to generate schedule");
  }
  return (await res.json()) as {
    blocks: ScheduleBlock[];
    analysis: ScheduleAnalysis[];
    rag_answer: string | null;
  };
}

export async function chatAdjust(input: {
  syllabus: ParsedSyllabus;
  topics: Topic[];
  message: string;
}) {
  const res = await fetch(`${API_BASE}/api/chat-adjust`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to adjust settings via chat");
  }
  return (await res.json()) as {
    reply: string;
    updates: ChatAdjustUpdate[];
  };
}
