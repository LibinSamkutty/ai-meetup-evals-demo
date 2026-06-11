# database.py — Operation Blackout
import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = "data/eval_results.db"


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db() -> None:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp            TEXT,
                session_id           TEXT,
                persona_key          TEXT,
                persona_display_name TEXT,
                question_id          TEXT,
                question_text        TEXT,
                segment_tag          TEXT,
                response_text        TEXT,
                context_chunks       TEXT,
                latency_ms           INTEGER,
                overall_verdict      TEXT,
                rationale            TEXT,
                dimensions_json      TEXT,
                hallucinated_claims  TEXT,
                rule_flags           TEXT,
                judges_json          TEXT
            )
            """
        )
        # Safe migration for databases that predate judges_json
        try:
            c.execute("ALTER TABLE eval_runs ADD COLUMN judges_json TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_eval_run(run: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO eval_runs (
                timestamp, session_id, persona_key, persona_display_name,
                question_id, question_text, segment_tag,
                response_text, context_chunks, latency_ms,
                overall_verdict, rationale, dimensions_json,
                hallucinated_claims, rule_flags, judges_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                run.get("session_id", ""),
                run.get("persona_key", ""),
                run.get("persona_display_name", ""),
                run.get("question_id", ""),
                run.get("question_text", ""),
                run.get("segment_tag", ""),
                run.get("response_text", ""),
                json.dumps(run.get("context_chunks", [])),
                run.get("latency_ms", 0),
                run.get("overall_verdict", ""),
                run.get("rationale", ""),
                json.dumps(run.get("dimensions", {})),
                json.dumps(run.get("hallucinated_claims", [])),
                json.dumps(run.get("rule_flags", [])),
                json.dumps(run.get("judges", {})),
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _parse_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    for field in ("context_chunks", "hallucinated_claims", "rule_flags"):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    if d.get("dimensions_json"):
        try:
            d["dimensions"] = json.loads(d["dimensions_json"])
        except (json.JSONDecodeError, TypeError):
            d["dimensions"] = {}
    else:
        d["dimensions"] = {}
    if d.get("judges_json"):
        try:
            d["judges"] = json.loads(d["judges_json"])
        except (json.JSONDecodeError, TypeError):
            d["judges"] = {}
    else:
        d["judges"] = {}
    return d


def get_all_runs(session_id: str | None = None) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        c = conn.cursor()
        if session_id:
            c.execute(
                "SELECT * FROM eval_runs WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,),
            )
        else:
            c.execute("SELECT * FROM eval_runs ORDER BY timestamp DESC")
        rows = [_parse_row(r) for r in c.fetchall()]
    finally:
        conn.close()
    return rows


def get_prior_responses(
    question_id: str,
    persona_key: str,
    limit: int = 2,
) -> list[str]:
    """Return up to `limit` prior response texts for this question/persona, newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        c = conn.cursor()
        c.execute(
            """
            SELECT response_text FROM eval_runs
            WHERE question_id = ? AND persona_key = ?
              AND response_text != ''
            ORDER BY timestamp DESC LIMIT ?
            """,
            (question_id, persona_key, limit),
        )
        rows = c.fetchall()
        return [r["response_text"] for r in rows]
    finally:
        conn.close()
