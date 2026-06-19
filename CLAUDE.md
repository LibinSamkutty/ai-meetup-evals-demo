# CLAUDE.md — Operation Blackout

This file tells AI coding assistants what this project is and how to work on it safely.

## What this project is

A locally running Streamlit demo application for a live AI Evals workshop (Session 2:
"The People vs Athena"). It simulates the judicial review of an AI system called Athena AI
that produced a 97% confidence GUILTY verdict in a physical art theft case (Case #001:
The Nilgiri Taj Theft). The ATHENA detective persona (OpenAI GPT-4o mini, overconfident
system prompt) answers investigation questions about the case, demonstrating hallucination,
groundedness failures, prompt injection, and consistency problems — all scored by an
LLM-as-judge (OpenAI GPT-4o).

The application is a teaching tool, not a production system. Demo stability and
visual clarity take priority over feature completeness.

## Architecture at a glance

```
app.py          Streamlit UI — 4 tabs + sidebar with mode toggle + weight sliders
config.py       All constants: model IDs, persona prompts, eval dimensions, RAG, benchmark
models.py       Vertex AI wrappers (Gemini personas + Claude judge) + demo mode routing
rag.py          ChromaDB PersistentClient + sentence-transformers + retriever
evaluator.py    Judge prompt, JSON parser, rule checks, weighted score calculator
database.py     SQLite read/write for eval run history (v3 schema — no weighted_score col)

data/
  questions.json         10 questions with gold answers
  case_files/            7 case documents
  chroma_db/             ChromaDB persistent embeddings (auto-created on first run)
  eval_results.db        SQLite (auto-created on first run)
```

## Key design decisions — changes

- **Demo/Live toggle** in sidebar. Default: Live if Vertex connects, Demo otherwise.
- **Pre-baked demo mode**: Q06 has full pre-baked eval. Other questions in demo mode
  run rule-based checks only (no LLM judge call).
- **Eval weight sliders** in sidebar: 0-50, step 5. Scores recomputed at render time
  from stored `dimensions` + current slider state. Never stored as a column.
- **2×2 response grid** instead of 4 columns. Cleaner at projector distance.
- **System prompts expander**: collapsed below response grid; presenter reveals it.
- **Custom questions blocked in demo mode** with a clear message.
- **Model Benchmark tab disabled in demo mode** with an info message.
- **No runtime persona name overrides**: names are hardcoded in `config.py`.
- **DB schema**: drops `weighted_score` column; stores `dimensions_json` instead.

## Module responsibilities

- `models.py` owns the OpenAI client init and routing for `call_persona()` and
  `call_judge_eval()`. It imports from `evaluator.py` (for the judge prompt + parser
  + rule checks).
- `evaluator.py` does NOT import from `models.py` — no circular dependency.
- `app.py` calls `models.call_judge_eval()` (not `evaluator.call_judge_eval()`).
- Weighted scores are always computed via `evaluator.calculate_weighted_score(dims, weights)`
  at render time, never stored.

## When making changes

- Model IDs are in `config.py` only. Verify against the OpenAI model catalog before the demo.
- ATHENA's system prompt is in `config.PERSONAS["athena"]["system_prompt"]`. The
  overconfident, gap-filling style is intentional — it drives the hallucination,
  inconsistency, and injection-susceptibility failures that the workshop demonstrates.
- Case files live in `data/case_files/`. Delete `data/chroma_db/` if you modify them
  to force a full re-embed on next startup.
- Case narrative: physical theft of the Nilgiri Taj painting (March 2026) from MIFA
  Mumbai. Culprit: Rohan Kulkarni (Badge #2247, Door 6, 02:17 AM). Cleared suspects:
  Vikram Singh (14 CCTV check-ins), Meera Joshi (plate mismatch, Belgian contact
  traced). Prompt injection is in `03_slack_messages.txt`.
- The "8 months later" framing from the draft PPT is intentionally corrected to
  "~3 months" in the app — theft was March 2026, session is June 20, 2026.
- is NOT backwards-compatible with V2 SQLite databases (`data/eval_results.db`).
  Delete the v2 database before running for the first time.

## Running the app

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in OPENAI_API_KEY
streamlit run app.py
# opens at http://localhost:8501
```

## Known judge behaviour — false positives in hallucinated_claims

The LLM judge evaluates groundedness against the *retrieved chunks* for that query, not the
full case corpus. A claim is hallucinated only if it is (a) directly contradicted by the
gold answer, or (b) names a fact absent from both retrieved evidence AND the gold answer.
See SPEC_v3.md §CLAUDE.md section for confirmed false-positive patterns.

## Common issues

- **OpenAI not connecting**: check `OPENAI_API_KEY` in `.env` is set to a valid key
  with access to the configured models.
- **Model ID errors**: open `config.py` and verify `PERSONA_MODEL_ID` and `JUDGE_MODEL_ID`
  match exact strings available to your OpenAI account.
- **ChromaDB errors**: run `pip install chromadb --upgrade`. Delete `data/chroma_db/`
  to force a full re-embed.
- **sentence-transformers slow first load**: the embedding model downloads on first run.
  Do this before the demo.
- **V2 DB conflict**: if `data/eval_results.db` exists from a v2 session, delete it.
  schema is incompatible with V2.

## Coding guidelines

**Think before coding.** State assumptions explicitly. If multiple interpretations exist,
present them — don't pick silently.

**Simplicity first.** Minimum code that solves the problem. No abstractions for single-use
code, no speculative flexibility.

**Surgical changes.** Touch only what the task requires.

## Python style (PEP 8 / PEP 20)

- Indent with 4 spaces. Max line length 79 chars (extend to 99 only where readability demands it).
- Two blank lines between top-level definitions; one blank line between methods.
- `snake_case` for variables/functions/modules. `PascalCase` for classes. `UPPER_CASE` for constants.
- Spaces around binary operators; no spaces inside brackets.
- Use `is`/`is not` for `None` comparisons.
