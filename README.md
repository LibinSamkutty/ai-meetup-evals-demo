# Operation Blackout —

A locally running Streamlit demo for a live AI Evals workshop. The ATHENA detective persona
— OpenAI GPT-4o mini, overconfident system prompt — investigates the Nilgiri Taj theft case.
An LLM-as-judge (OpenAI GPT-4o) scores every response across five dimensions.
The audience sees hallucination, groundedness failure, prompt injection, and non-determinism
made measurable in real time.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY

# 4. Run
streamlit run app.py
# Opens at http://localhost:8501
```

The first run downloads the sentence-transformers embedding model (~90 MB) and builds the
ChromaDB index. Subsequent starts reuse the persisted index in `data/chroma_db/`.

---

## Demo vs Live Mode

The app has two modes, toggled from the sidebar:

| | Demo Mode 🔴 | Live Mode 🟢 |
|---|---|---|
| Startup default | If OpenAI fails to connect | If OpenAI connects |
| Reveal Eval Scores (Q06) | Pre-baked eval from JSON | LLM judge + rule checks |
| Reveal Eval Scores (other Q) | Rule-based checks only | LLM judge + rule checks |
| Consistency Test | Pre-baked variants (Q06/ATHENA only) | 3 live OpenAI calls |
| Custom questions | Blocked with message | Allowed |
| Model Benchmark tab | Disabled | Available |

---

## Project Structure

```
app.py              Streamlit UI — 4 tabs + sidebar
config.py           Constants: model IDs, personas, eval dimensions, RAG, benchmark
models.py           OpenAI wrappers + demo mode routing
rag.py              ChromaDB PersistentClient + sentence-transformers retriever
evaluator.py        Judge prompt template, JSON parser, rule checks, weighted scorer
database.py         SQLite (v3 schema — no weighted_score column; stores dimensions_json)

data/
  questions.json         10 questions with gold answers and segment tags
  case_files/            7 case documents — the AI detectives' only evidence source
  chroma_db/             Auto-created on first run; delete to force re-embed
  eval_results.db        Auto-created on first run; delete when switching from v2

docs/
  SPEC_v3.md             Full specification and decision log
  demo.md                Presenter flow guide — segment-by-segment talking points
  PROJECT_OVERVIEW.md    Architecture, personas, teaching point map

.env.example        Environment variable template
requirements.txt    Python dependencies
CLAUDE.md           AI assistant instructions for this codebase
```

---

## Environment Variables

```bash
# .env (copy from .env.example)
OPENAI_API_KEY=your-openai-api-key   # required for Live Mode
```

---

## Eval Weight Sliders

The sidebar has sliders for each of the five eval dimensions (0-50, step 5). Moving a slider
**immediately recomputes all scores** from stored dimension verdicts — no new API calls.
Scores are never stored; they are always derived at render time from raw `dimensions_json`
and the current slider state.

Default weights: Correctness 30%, Groundedness 30%, Faithfulness 20%, Relevance 10%, Safety 10%.

---

## Detective

| Key | Name | Colour | Style |
|---|---|---|---|
| athena | ATHENA ⚖️ | Red | Aggressive · Overconfident · High hallucination risk |

ATHENA uses OpenAI GPT-4o mini. Its overconfident system prompt is the workshop's
demonstration object: the same model with a different prompt produces measurably
different failure modes.

---

## Key Teaching Moments

| What to show | Question | Feature |
|---|---|---|
| Same model, different prompts | Q06 | Run Investigation |
| Eval reveal with ATHENA's fabrications | Q06 | Reveal Eval Scores |
| Prompt injection via Slack HTML comment | Q07 | Injection warning banner |
| Hallucination trap — no answer exists | Q02, Q04, Q10 | Rule flag: false_certainty |
| Faithfulness — must include both sides | Q05, Q11 | Judge faithfulness dim |
| Retrieval gap → judge false positive | Q03 | Retrieval Gap warning |
| Non-determinism across 3 runs | Q06 (live) | Consistency Test |
| Live weight change changes leaderboard | Any | Sidebar sliders |
| Only difference is the system prompt | Any | System Prompts expander |
| Model capability vs prompt engineering | Any | Model Benchmark tab |

---

## Common Issues

**OpenAI not connecting**
Check `OPENAI_API_KEY` in `.env` is set to a valid key with access to the configured
models. The app falls back to Demo Mode automatically.

**Model ID errors**
Open `config.py` and verify `PERSONA_MODEL_ID` and `JUDGE_MODEL_ID` match exact strings
available to your OpenAI account.

**ChromaDB errors**
Run `pip install chromadb --upgrade`. Delete `data/chroma_db/` to force a full re-embed
if you've modified the case files.

**Slow first startup**
The sentence-transformers embedding model downloads on first run (~90 MB). Run the app
once on your presentation machine before the workshop.

**V2 database conflict**
If `data/eval_results.db` exists from a v2 session, delete it. uses a different schema
that drops the `weighted_score` column.