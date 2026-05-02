"use client";

import { useMemo, useState } from "react";
import {
  chatAdjust,
  ChatMessage,
  generateSchedule,
  parseSyllabus,
  ParsedSyllabus,
  ParseQuality,
  ScheduleAnalysis,
  ScheduleBlock,
  Topic,
} from "../lib/api";

const weekdayMap: Record<string, number> = {
  Monday: 0,
  Tuesday: 1,
  Wednesday: 2,
  Thursday: 3,
  Friday: 4,
  Saturday: 5,
  Sunday: 6,
};

function toTopicRows(syllabus: ParsedSyllabus): Topic[] {
  return syllabus.topics.map((topic) => ({
    title: topic.title,
    priority: Number(topic.weightage_percent ?? 1),
    target_minutes: Number(topic.estimated_hours ? topic.estimated_hours * 60 : 0),
    difficulty: 1,
    has_deadline: false,
    deadline: new Date(Date.now() + 14 * 24 * 3600 * 1000).toISOString().slice(0, 10),
  }));
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [useLlm, setUseLlm] = useState(true);
  const [optimizerMode, setOptimizerMode] = useState<"cp_sat" | "greedy">("cp_sat");
  const [includeReviews, setIncludeReviews] = useState(true);
  const [strictMode, setStrictMode] = useState(true);
  const [noStudy, setNoStudy] = useState<string[]>([]);
  const [syllabus, setSyllabus] = useState<ParsedSyllabus | null>(null);
  const [parseQuality, setParseQuality] = useState<ParseQuality | null>(null);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [blocks, setBlocks] = useState<ScheduleBlock[]>([]);
  const [analysis, setAnalysis] = useState<ScheduleAnalysis[]>([]);
  const [ragAnswer, setRagAnswer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Ask changes like: increase priority of Unit 3, set 240 minutes for DBMS, set OS deadline to 2026-06-01.",
    },
  ]);
  const [error, setError] = useState<string | null>(null);

  const agenda = useMemo(() => {
    const grouped: Record<string, ScheduleBlock[]> = {};
    for (const block of blocks) {
      if (!grouped[block.date]) grouped[block.date] = [];
      grouped[block.date].push(block);
    }
    return Object.entries(grouped).sort((a, b) => a[0].localeCompare(b[0]));
  }, [blocks]);

  async function onParse() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await parseSyllabus(file, useLlm);
      setSyllabus(res.syllabus);
      setParseQuality(res.parse_quality);
      setTopics(toTopicRows(res.syllabus));
      setBlocks([]);
      setAnalysis([]);
      setRagAnswer(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Parse failed");
    } finally {
      setLoading(false);
    }
  }

  async function runGenerate(topicRows: Topic[]) {
    if (!syllabus) return;
    const res = await generateSchedule({
      syllabus,
      topics: topicRows,
      optimizer_mode: optimizerMode,
      include_reviews: includeReviews,
      strict_mode: strictMode,
      query: query.trim() || undefined,
      no_study_weekdays: noStudy.map((d) => weekdayMap[d]),
    });
    setBlocks(res.blocks);
    setAnalysis(res.analysis);
    setRagAnswer(res.rag_answer);
  }

  async function onGenerate() {
    if (!syllabus) return;
    setLoading(true);
    setError(null);
    try {
      await runGenerate(topics);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Schedule generation failed");
    } finally {
      setLoading(false);
    }
  }

  async function onChatSend() {
    if (!syllabus || !chatInput.trim()) return;
    const userMessage = chatInput.trim();
    setChatInput("");
    setChatLoading(true);
    setError(null);
    setChatMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    try {
      const adjusted = await chatAdjust({
        syllabus,
        topics,
        message: userMessage,
      });
      const updatedTopics = topics.map((topic) => {
        const patch = adjusted.updates.find((item) => item.title === topic.title);
        if (!patch) return topic;
        return {
          ...topic,
          priority: patch.priority ?? topic.priority,
          target_minutes: patch.target_minutes ?? topic.target_minutes,
          difficulty: patch.difficulty ?? topic.difficulty,
          has_deadline: patch.has_deadline ?? topic.has_deadline,
          deadline: patch.deadline ?? topic.deadline,
        };
      });
      setTopics(updatedTopics);
      await runGenerate(updatedTopics);
      setChatMessages((prev) => [...prev, { role: "assistant", content: adjusted.reply }]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Chat adjustment failed";
      setError(msg);
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: `I could not apply that: ${msg}` },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  function updateTopic(index: number, key: keyof Topic, value: string | number | boolean) {
    setTopics((prev) =>
      prev.map((topic, i) => (i === index ? { ...topic, [key]: value } : topic)),
    );
  }

  return (
    <main className="container">
      <h1>24/7 Personalized Teaching Assistant</h1>
      <p className="subtitle">
        Next.js dashboard for syllabus parsing, priority tuning, and schedule generation.
      </p>

      <div className="layout">
        <div className="planner-pane">
      <section className="card">
        <h2>Upload and Parse</h2>
        <div className="grid">
          <div>
            <label>Upload syllabus PDF</label>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </div>
          <div>
            <label>Optional RAG question</label>
            <input value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
        </div>
        <div className="row">
          <label>
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
            />
            Use LLM parsing
          </label>
          <button disabled={!file || loading} onClick={onParse}>
            {loading ? "Parsing..." : "Parse Syllabus"}
          </button>
        </div>
      </section>

      {parseQuality && (
        <section className="card metrics">
          <div>
            <p>Topics</p>
            <strong>{topics.length}</strong>
          </div>
          <div>
            <p>Raw text lines</p>
            <strong>{parseQuality.lines}</strong>
          </div>
          <div>
            <p>Heading-like lines</p>
            <strong>{parseQuality.heading_like_lines}</strong>
          </div>
          <div>
            <p>Parse quality</p>
            <strong>{parseQuality.score}/100</strong>
          </div>
        </section>
      )}

      {syllabus && (
        <section className="card">
          <h2>Priority Settings</h2>
          <div className="row controls">
            <label>
              Optimizer
              <select
                value={optimizerMode}
                onChange={(e) => setOptimizerMode(e.target.value as "cp_sat" | "greedy")}
              >
                <option value="cp_sat">cp_sat</option>
                <option value="greedy">greedy</option>
              </select>
            </label>
            <label>
              <input
                type="checkbox"
                checked={includeReviews}
                onChange={(e) => setIncludeReviews(e.target.checked)}
              />
              Include reviews
            </label>
            <label>
              <input
                type="checkbox"
                checked={strictMode}
                onChange={(e) => setStrictMode(e.target.checked)}
              />
              Strict mode
            </label>
          </div>

          <label>No-study weekdays</label>
          <div className="weekday-row">
            {Object.keys(weekdayMap).map((day) => (
              <label key={day}>
                <input
                  type="checkbox"
                  checked={noStudy.includes(day)}
                  onChange={(e) =>
                    setNoStudy((prev) =>
                      e.target.checked ? [...prev, day] : prev.filter((d) => d !== day),
                    )
                  }
                />
                {day}
              </label>
            ))}
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Topic</th>
                  <th>Priority</th>
                  <th>Target Minutes</th>
                  <th>Difficulty</th>
                  <th>Has Deadline</th>
                  <th>Deadline</th>
                </tr>
              </thead>
              <tbody>
                {topics.map((topic, idx) => (
                  <tr key={topic.title}>
                    <td>{topic.title}</td>
                    <td>
                      <input
                        type="number"
                        step={0.1}
                        min={0.1}
                        max={100}
                        value={topic.priority}
                        onChange={(e) => updateTopic(idx, "priority", Number(e.target.value))}
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        step={15}
                        min={0}
                        max={5000}
                        value={topic.target_minutes}
                        onChange={(e) =>
                          updateTopic(idx, "target_minutes", Number(e.target.value))
                        }
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        step={0.1}
                        min={0.5}
                        max={3}
                        value={topic.difficulty}
                        onChange={(e) => updateTopic(idx, "difficulty", Number(e.target.value))}
                      />
                    </td>
                    <td>
                      <input
                        type="checkbox"
                        checked={topic.has_deadline}
                        onChange={(e) => updateTopic(idx, "has_deadline", e.target.checked)}
                      />
                    </td>
                    <td>
                      <input
                        type="date"
                        disabled={!topic.has_deadline}
                        value={topic.deadline}
                        onChange={(e) => updateTopic(idx, "deadline", e.target.value)}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <button disabled={loading} onClick={onGenerate}>
            {loading ? "Generating..." : "Generate Schedule"}
          </button>
        </section>
      )}

      {!!ragAnswer && (
        <section className="card">
          <h2>RAG Answer</h2>
          <p>{ragAnswer}</p>
        </section>
      )}

      {blocks.length > 0 && (
        <section className="card">
          <h2>Daily Agenda</h2>
          {agenda.map(([day, dayBlocks]) => (
            <details key={day}>
              <summary>
                {day} - {dayBlocks.reduce((acc, b) => acc + b.duration_minutes, 0)} minutes
              </summary>
              <ul>
                {dayBlocks.map((block, i) => (
                  <li key={`${day}-${i}`}>
                    {block.start_time.slice(0, 5)} ({block.duration_minutes}m) [{block.type}]{" "}
                    {block.topic}
                  </li>
                ))}
              </ul>
            </details>
          ))}
        </section>
      )}

      {analysis.length > 0 && (
        <section className="card">
          <h2>Priority Analysis</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Topic</th>
                  <th>Target</th>
                  <th>Planned</th>
                  <th>Coverage %</th>
                  <th>Strict Score</th>
                  <th>Overdue Minutes</th>
                </tr>
              </thead>
              <tbody>
                {analysis.map((row) => (
                  <tr key={row.topic}>
                    <td>{row.topic}</td>
                    <td>{row.target_minutes}</td>
                    <td>{row.planned_minutes}</td>
                    <td>{row.coverage_pct ?? "-"}</td>
                    <td>{row.strict_score}</td>
                    <td>{row.overdue_minutes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
        </div>

        <aside className="chat-pane">
          <section className="card chat-card">
            <h2>Planner Chat</h2>
            <p className="chat-subtitle">
              Ask for real-time updates to priorities, minutes, and deadlines.
            </p>
            <div className="chat-log">
              {chatMessages.map((msg, idx) => (
                <div
                  key={`${msg.role}-${idx}`}
                  className={`chat-bubble ${msg.role === "user" ? "chat-user" : "chat-assistant"}`}
                >
                  {msg.content}
                </div>
              ))}
              {chatLoading && <div className="chat-bubble chat-assistant">Applying updates...</div>}
            </div>
            <div className="chat-input-row">
              <textarea
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Increase priority of Unit 2 and set deadline to 2026-06-10"
                rows={3}
              />
              <button disabled={!syllabus || chatLoading || !chatInput.trim()} onClick={onChatSend}>
                {chatLoading ? "Working..." : "Send"}
              </button>
            </div>
          </section>
        </aside>
      </div>

      {error && <p className="error">{error}</p>}
    </main>
  );
}
