# 24/7 Personalized Teaching Assistant

An AI-powered study scheduler that turns PDF syllabi into smart, personalized study plans using **Spaced Repetition**, **RAG**, and **optimization**. It respects your deep work windows and syncs directly to Google Calendar.

Built for students (especially IIT-H and similar institutions) to reduce pre-exam stress and improve academic outcomes.

![Project Banner](https://via.placeholder.com/800x200?text=Personalized+Teaching+Assistant)  
*(Replace with a nice screenshot of your UI/calendar view)*

## ✨ Features

- Deep parsing of syllabus PDFs to extract topics, weightage, learning objectives, and exam dates
- "Chat with My Syllabus" interface powered by **RAG** (LangChain)
- Intelligent study block allocation using **Google OR-Tools CP-SAT** (with greedy fallback)
- Spaced Repetition scheduling for better long-term retention
- Automatic sync to Google Calendar with reminders
- Progress tracking and adaptive re-optimization

## 🛠️ Tech Stack

- **Backend**: Python + FastAPI (or Flask/Streamlit for MVP)
- **AI/RAG**: LangChain, OpenAI/Gemini/Groq (or local LLMs)
- **PDF Parsing**: pypdf + Unstructured.io / LlamaParse
- **Optimization**: Google OR-Tools (CP-SAT) + Custom Greedy Heuristic
- **Vector Store**: FAISS or Chroma
- **Calendar**: Google Calendar API
- **Storage**: SQLite / PostgreSQL
- **Deployment**: Docker (recommended)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Step-by-Step Development Approach](#step-by-step-development-approach)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Challenges & Solutions](#challenges--solutions)
- [Contributing](#contributing)
- [License](#license)

## 📖 Project Overview

Students often struggle with time management and prioritizing topics from dense syllabi.  
This tool solves that by:
1. Automatically understanding your syllabus
2. Building an optimized weekly study schedule
3. Incorporating Spaced Repetition
4. Syncing everything to your calendar

Result: Smarter study habits and better academic performance.

## 🚀 Step-by-Step Development Approach

Follow this structured path to build the project progressively:

### Phase 1: Planning & Setup
- Define requirements and user flow
- Choose tech stack
- Set up Git repository, virtual environment, and basic project structure
- Get API keys (Google Cloud for Calendar + OR-Tools, OpenAI/Gemini)

### Phase 2: Syllabus Parsing
- Load and extract text from PDFs
- Use LLM + structured output (Pydantic) to extract topics, weightage, exam dates
- Save parsed data as JSON and embeddings

### Phase 3: RAG Interface ("Chat with My Syllabus")
- Implement document loading, chunking, and embedding
- Build retrieval chain with LangChain
- Create a simple chat UI (Streamlit or Gradio)
- Ensure answers are grounded in the syllabus

### Phase 4: Optimization Engine
- Implement **greedy heuristic** as a fast baseline (see `scheduler/greedy.py`)
- Build full **CP-SAT model** using Google OR-Tools for constraint optimization
- Add Spaced Repetition logic (initial study + review intervals)
- Support deep work windows, deadlines, and daily limits

### Phase 5: Calendar Integration
- Set up Google Calendar API with OAuth
- Generate and sync study events (with color-coding and reminders)
- Handle conflicts and progress updates

### Phase 6: Full Pipeline & UI
- Connect all components: Upload → Parse → Chat → Optimize → Sync
- Add progress tracking (mark sessions complete → re-optimize)
- Build a clean dashboard

### Phase 7: Testing, Polish & Deployment
- Test with multiple real syllabi
- Add error handling and logging
- Containerize with Docker
- Deploy (Vercel, Render, or cloud VM)

## 🛠️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/24-7-personalized-teaching-assistant.git
cd 24-7-personalized-teaching-assistant

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys