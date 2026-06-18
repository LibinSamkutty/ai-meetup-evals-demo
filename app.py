# app.py — Operation Blackout
import inspect
import json
import logging
import re
import uuid
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="huggingface_hub")

import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import streamlit as st  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()

import config
import database
import evaluator
import models
import rag

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Case #002: The People vs Athena",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0f0f0f; color: #cccccc; }

    section[data-testid="stSidebar"] {
        background: #161616;
        border-right: 1px solid #2a2a2a;
        overflow-x: hidden;
    }

    section[data-testid="stSidebar"] .stSlider {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }

    section[data-testid="stSidebar"] .stSlider label {
        margin-bottom: 0rem;
    }

    section[data-testid="stSidebar"] .block-container,
    section[data-testid="stSidebar"] > div > div > div {
        gap: 0.25rem;
    }

    section[data-testid="stSidebar"] .stButton {
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
        font-size: 0.9rem;
    }

    .detective-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .detective-name {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .detective-style {
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 12px;
    }

    .response-text {
        font-size: 0.9rem;
        color: #cccccc;
        line-height: 1.6;
    }

    .verdict-pass {
        background: #1a4731;
        color: #4ade80;
        padding: 2px 10px;
        border-radius: 4px;
        font-weight: 700;
    }

    .verdict-fail {
        background: #4a1515;
        color: #f87171;
        padding: 2px 10px;
        border-radius: 4px;
        font-weight: 700;
    }

    .injection-warning {
        background: #2a0a00;
        border: 1px solid #e8540a;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.82rem;
        color: #ffaa77;
    }

    .case-file-content {
        background: #0d0d0d;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 14px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #ccc;
        white-space: pre-wrap;
    }

    .score-bar-track { background: #222; border-radius: 4px; height: 6px; }
    .flag-critical   { color: #f87171; font-weight: 700; }
    .flag-major      { color: #fb923c; font-weight: 600; }
    .flag-minor      { color: #facc15; }

    hr { border-color: #2a2a2a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8].upper()
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "consistency_results" not in st.session_state:
    st.session_state.consistency_results = None
if "golden_annotations" not in st.session_state:
    st.session_state.golden_annotations = {}
if "golden_locked" not in st.session_state:
    st.session_state.golden_locked = False
if "golden_dataset_version" not in st.session_state:
    st.session_state.golden_dataset_version = 1
if "hitl_overrides" not in st.session_state:
    st.session_state.hitl_overrides = {}
if "show_taxonomy" not in st.session_state:
    st.session_state.show_taxonomy = False
if "model_params" not in st.session_state:
    st.session_state.model_params = {
        "temperature": 1.5,
        "top_k": 40,
        "top_p": 0.95,
    }
if "original_persona_prompts" not in st.session_state:
    st.session_state.original_persona_prompts = {
        k: v["system_prompt"] for k, v in config.PERSONAS.items()
    }
if "original_judge_prompt" not in st.session_state:
    st.session_state.original_judge_prompt = (
        evaluator.JUDGE_PROMPT_TEMPLATE
    )
for _pk in config.PERSONAS:
    if f"editing_persona_{_pk}" not in st.session_state:
        st.session_state[f"editing_persona_{_pk}"] = False
if "editing_judge_prompt" not in st.session_state:
    st.session_state.editing_judge_prompt = False
if "original_vera_judge_prompt" not in st.session_state:
    st.session_state.original_vera_judge_prompt = (
        evaluator.VERA_JUDGE_PROMPT_TEMPLATE
    )
if "editing_vera_judge_prompt" not in st.session_state:
    st.session_state.editing_vera_judge_prompt = False
if "original_axiom_checks" not in st.session_state:
    st.session_state.original_axiom_checks = {}
    st.session_state.axiom_check_sources = {}
    for _qid, _fn in evaluator._PER_QUESTION_CHECKS.items():
        try:
            _src = inspect.getsource(_fn)
        except OSError:
            _src = f"# Source unavailable for {_qid}"
        st.session_state.original_axiom_checks[_qid] = _src
        st.session_state.axiom_check_sources[_qid] = _src
if "editing_axiom_check" not in st.session_state:
    st.session_state.editing_axiom_check = None

# ---------------------------------------------------------------------------
# One-time startup: DB init + RAG index + Vertex probe
# ---------------------------------------------------------------------------

database.init_db()


@st.cache_resource(show_spinner="Starting up…")
def _startup():
    vertex_err = ""
    try:
        models.init_vertex()
        vertex_ok = True
    except Exception as exc:
        vertex_ok = False
        vertex_err = str(exc)
    chunk_count, built_fresh = rag.build_index()
    return chunk_count, built_fresh, vertex_ok, vertex_err


chunk_count, index_built_fresh, vertex_ok, vertex_err = _startup()

if "mode" not in st.session_state:
    st.session_state.mode = "live"


@st.cache_data(show_spinner=False)
def _load_questions():
    with open("data/questions.json", "r") as fh:
        return json.load(fh)


all_questions = _load_questions()


# ---------------------------------------------------------------------------
# Golden dataset helpers
# ---------------------------------------------------------------------------

def _effective_gold(q: dict) -> str:
    """Return the annotation-edited gold answer if one exists, else the original."""
    return (
        st.session_state.golden_annotations.get(q["id"], {}).get("gold_answer")
        or q["gold_answer"]
    )


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Question selector (shared by Review Board + Benchmark tabs)
# ---------------------------------------------------------------------------

def _render_question_selector(prefix: str, questions: list[dict]) -> dict | None:
    text_key = f"{prefix}_question_text"
    preset_key = f"{prefix}_preset_select"
    prev_preset_key = f"_{prefix}_prev_preset"

    preset_labels = {q["id"]: q["question"] for q in questions}
    preset_qid = st.selectbox(
        "Load preset question",
        options=[""] + [q["id"] for q in questions],
        format_func=lambda x: "— pick a preset to load —" if x == "" else preset_labels[x],
        key=preset_key,
    )
    if preset_qid != st.session_state.get(prev_preset_key):
        if preset_qid:
            match = next((q for q in questions if q["id"] == preset_qid), None)
            if match:
                st.session_state[text_key] = match["question"]
        st.session_state[prev_preset_key] = preset_qid

    question_text = st.text_area(
        "Question",
        key=text_key,
        placeholder="Type your own question, or load a preset above…",
        height=80,
    )

    if not question_text.strip():
        return None

    exact = next(
        (q for q in questions if q["question"].strip() == question_text.strip()),
        None,
    )
    if exact:
        return exact

    return {
        "id": "custom",
        "question": question_text.strip(),
        "gold_answer": "(No gold answer — custom question)",
        "eval_intent": "Custom question entered live.",
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_consist, tab_files, tab_invest, tab_dash, tab_golden = st.tabs([
    "🔁 Consistency Test",
    "📁 Case Files",
    "🔍 Review Board",
    "📊 Eval Dashboard",
    "📋 Golden Dataset Lab",
])


# ===========================================================================
# TAB 0 — GOLDEN DATASET LAB
# ===========================================================================

with tab_golden:
    st.markdown("## 📋 Golden Dataset")

    if st.session_state.golden_locked:
        st.success(
            f"✅ Golden Dataset "
            f"**v{st.session_state.golden_dataset_version}** is locked. "
            f"{len(st.session_state.golden_annotations)} questions annotated."
        )
        _locked_rows = []
        for _q in all_questions:
            _ann = st.session_state.golden_annotations.get(
                _q["id"], {}
            )
            _locked_rows.append({
                "input": _q["question"],
                "gold_answer": _ann.get(
                    "gold_answer", _q["gold_answer"]
                ),
            })
        st.dataframe(
            pd.DataFrame(_locked_rows),
            use_container_width=True,
            hide_index=True,
        )
        _unlock_col, _ = st.columns([1, 3])
        with _unlock_col:
            if st.button(
                "🔓 Unlock to Edit", key="btn_unlock_golden"
            ):
                st.session_state.golden_locked = False
                st.rerun()
    else:
        _annotator_name = ""

        for _q in all_questions:
            _existing = st.session_state.golden_annotations.get(
                _q["id"], {}
            )
            with st.expander(_q["question"], expanded=False):
                _gold = st.text_area(
                    "Gold answer (expected output)",
                    value=_existing.get("gold_answer", _q["gold_answer"]),
                    height=100,
                    key=f"golden_gold_{_q['id']}",
                )

                st.session_state.golden_annotations[_q["id"]] = {
                    "severity": _existing.get("severity", "Major"),
                    "notes": _existing.get(
                        "notes", _q.get("eval_intent", "")
                    ),
                    "gold_answer": _gold,
                    "annotator": _annotator_name,
                }

        st.divider()
        _lock_col, _ = st.columns([1, 3])
        with _lock_col:
            if st.button(
                "🔒 Lock Dataset v1",
                type="primary",
                use_container_width=True,
                key="btn_lock_golden",
            ):
                for _q in all_questions:
                    if _q["id"] not in st.session_state.golden_annotations:
                        st.session_state.golden_annotations[_q["id"]] = {
                            "severity": "Major",
                            "notes": _q.get("eval_intent", ""),
                            "annotator": _annotator_name or "Workshop",
                        }
                st.session_state.golden_locked = True
                st.session_state.golden_dataset_version = 1
                st.rerun()


# ===========================================================================
# TAB 1 — INVESTIGATION ROOM
# ===========================================================================

_JUDGE_META = {
    "axiom": {
        "display_name": "AXIOM",
        "icon": "⚙️",
        "color": config.EVALUATORS["axiom"]["color"],
        "subtitle": "Code Assertions — deterministic per-question rules",
    },
    "nova": {
        "display_name": "NOVA",
        "icon": "🧠",
        "color": config.EVALUATORS["nova"]["color"],
        "subtitle": "LLM Judge — no ground truth",
    },
    "vera": {
        "display_name": "VERA",
        "icon": "🔬",
        "color": config.EVALUATORS["vera"]["color"],
        "subtitle": "LLM Judge (with ground truth)",
    },
}

with tab_invest:
    st.markdown("## 🔍 Review Board")

    selected_q = _render_question_selector("invest", all_questions)
    if selected_q is None:
        st.warning("Enter a question to investigate.")

    if selected_q is not None:
        is_custom = selected_q["id"] == "custom"

        st.divider()

        run_btn = st.button(
            "🚀 Run Investigation", type="primary", use_container_width=True
        )

        # --- RUN INVESTIGATION (single ATHENA) ---
        if run_btn:
            st.session_state.eval_results = None
            st.session_state.consistency_results = None

            with st.spinner("Retrieving evidence…"):
                chunks = rag.retrieve(
                    selected_q["question"], question_id=selected_q["id"]
                )
                context_str = rag.format_context(chunks)

            _athena = config.PERSONAS["athena"]
            with st.spinner("ATHENA AI investigating…"):
                _athena_result = models.call_persona(
                    system_prompt=_athena["system_prompt"],
                    query=selected_q["question"],
                    context=context_str,
                    persona_key="athena",
                    question_id=selected_q["id"],
                    temperature=0,
                )

            st.session_state.last_results = {
                "question": selected_q,
                "chunks": chunks,
                "context": context_str,
                "responses": {"athena": _athena_result},
            }
            st.rerun()

        # --- RESTORE RESPONSES ---
        if st.session_state.last_results:
            lr = st.session_state.last_results
            if lr["question"]["question"] == selected_q["question"]:

                st.divider()

                # Single ATHENA response card
                _athena_cfg = config.PERSONAS["athena"]
                _athena_resp = lr["responses"].get("athena", {})
                st.markdown(
                    f"<div class='detective-card' "
                    f"style='border-left:3px solid {_athena_cfg['color']}'>"
                    f"<div class='detective-name' "
                    f"style='color:{_athena_cfg['color']}'>"
                    f"{_athena_cfg['icon']} {_athena_cfg['display_name']}</div>"
                    f"<div class='detective-style'>"
                    f"{_athena_cfg['style']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if _athena_resp.get("error"):
                    st.error(f"⚠️ {_athena_resp['error']}")
                else:
                    st.markdown(_athena_resp.get("text", ""))
                    if _athena_resp.get("truncated"):
                        st.warning("⚠️ Response truncated (token limit)")

                st.divider()
                reveal_btn = st.button(
                    "🎯 Reveal Eval Scores",
                    type="primary",
                    use_container_width=True,
                )

                if reveal_btn:
                    _session_id = st.session_state.session_id
                    _athena_text = lr["responses"].get("athena", {}).get("text", "")
                    _qid = lr["question"]["id"]

                    if not _athena_text:
                        st.error("No ATHENA response to evaluate.")
                    else:
                        # AXIOM — local, no API call
                        with st.spinner("AXIOM: running rule checks…"):
                            _prior = database.get_prior_responses(
                                _qid, "athena", limit=2
                            )
                            _axiom_result = evaluator.run_per_question_rule_checks(
                                question_id=_qid,
                                response=_athena_text,
                                chunks=lr["chunks"],
                                prior_responses=_prior or None,
                            )

                        # NOVA + VERA — parallel API calls
                        _nova_result: dict = {}
                        _vera_result: dict = {}

                        _gold_answer = _effective_gold(lr["question"])

                        def _call_nova_judge() -> dict:
                            return models.call_judge_eval(
                                query=lr["question"]["question"],
                                response=_athena_text,
                                context=lr["context"],
                                persona_key="athena",
                                question_id=_qid,
                                chunks=lr["chunks"],
                                prior_responses=_prior or None,
                            )

                        def _call_vera_judge() -> dict:
                            return models.call_vera_eval(
                                query=lr["question"]["question"],
                                response=_athena_text,
                                context=lr["context"],
                                gold_answer=_gold_answer,
                                persona_key="athena",
                                question_id=_qid,
                                chunks=lr["chunks"],
                                prior_responses=_prior or None,
                            )

                        with st.spinner(
                            "NOVA + VERA: LLM judges evaluating…"
                        ):
                            with ThreadPoolExecutor(max_workers=2) as pool:
                                _fn = pool.submit(_call_nova_judge)
                                _fv = pool.submit(_call_vera_judge)
                                _nova_result = _fn.result()
                                _vera_result = _fv.result()

                        # Save to DB — one record, all 3 judge results
                        database.save_eval_run({
                            "session_id": _session_id,
                            "persona_key": "athena",
                            "persona_display_name": "ATHENA",
                            "question_id": _qid,
                            "question_text": lr["question"]["question"],
                            "response_text": _athena_text,
                            "context_chunks": lr["chunks"],
                            "latency_ms": lr["responses"].get(
                                "athena", {}
                            ).get("latency_ms", 0),
                            "overall_verdict": _nova_result.get(
                                "overall_verdict", ""
                            ),
                            "rationale": _nova_result.get("rationale", ""),
                            "dimensions": _nova_result.get("dimensions", {}),
                            "hallucinated_claims": _nova_result.get(
                                "hallucinated_claims", []
                            ),
                            "rule_flags": _axiom_result.get("rule_flags", []),
                            "judges": {
                                "vera": {
                                    "dimensions": _vera_result.get(
                                        "dimensions", {}
                                    ),
                                    "overall_verdict": _vera_result.get(
                                        "overall_verdict", ""
                                    ),
                                    "hallucinated_claims": _vera_result.get(
                                        "hallucinated_claims", []
                                    ),
                                    "rationale": _vera_result.get("rationale", ""),
                                },
                                "axiom": {
                                    "dimensions": _axiom_result.get(
                                        "dimensions", {}
                                    ),
                                    "overall_verdict": _axiom_result.get(
                                        "overall_verdict", ""
                                    ),
                                },
                            },
                        })

                        st.session_state.eval_results = {
                            "axiom": _axiom_result,
                            "nova": _nova_result,
                            "vera": _vera_result,
                        }

        # --- DISPLAY EVAL SCORES ---
        if (
            st.session_state.eval_results
            and st.session_state.last_results
            and st.session_state.last_results["question"]["question"]
            == selected_q["question"]
        ):
            er = st.session_state.eval_results
            st.divider()
            st.markdown("### 🎯 Eval Scores")
            st.caption(
                "**AXIOM** (Code Rules) · **NOVA** (LLM Judge, no ground truth) · "
                "**VERA** (LLM Judge, with gold answer)"
            )

            with st.expander("📋 Ground Truth Answer (given to VERA only)"):
                st.success(
                    _effective_gold(st.session_state.last_results["question"])
                )
            st.caption(
                "💡 **Teaching point:** NOVA and AXIOM had no access to this "
                "answer — a confident wrong conclusion can still pass their "
                "checklists. VERA uses it to verify factual correctness."
            )


            # Summary bar — 3 judges
            _sum_cols = st.columns(3)
            for _ji, (_jk, _jm) in enumerate(_JUDGE_META.items()):
                _jresult = er.get(_jk, {})
                _jdims = _jresult.get("dimensions", {})
                _jscore = evaluator.calculate_score(_jdims)
                _jverdict = _jresult.get("overall_verdict", "—")
                _jvc = (
                    "#22c55e" if _jverdict == "PASS"
                    else "#ef4444" if _jverdict == "FAIL"
                    else "#9ca3af"
                )
                with _sum_cols[_ji]:
                    st.markdown(
                        f"<div style='background:#1e1e2e;border:1px solid "
                        f"{_jm['color']}55;border-radius:8px;padding:10px 12px;"
                        f"text-align:center'>"
                        f"<div style='color:{_jm['color']};font-weight:700;"
                        f"font-size:0.85rem'>"
                        f"{_jm['icon']} {_jm['display_name']}</div>"
                        f"<div style='color:#666;font-size:0.72rem;"
                        f"margin-bottom:4px'>{_jm['subtitle']}</div>"
                        f"<div style='color:{_jvc};font-weight:700;"
                        f"font-size:1.2rem'>{_jverdict}</div>"
                        f"<div style='color:#e2e8f0;font-size:0.95rem'>"
                        f"{_jscore:.0f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown(
                "<div style='margin-top:12px'></div>", unsafe_allow_html=True
            )

            # 3-judge detail cards — one per judge
            _judge_cols = st.columns(3, gap="large")
            for _ji, (_jk, _jm) in enumerate(_JUDGE_META.items()):
                _jresult = er.get(_jk, {})
                _jdims = _jresult.get("dimensions", {})
                _jscore = evaluator.calculate_score(_jdims)
                _jverdict = _jresult.get("overall_verdict", "—")
                _jpass = _jverdict == "PASS"
                _jvc = "verdict-pass" if _jpass else "verdict-fail"

                with _judge_cols[_ji]:
                    with st.container(border=True):
                        st.markdown(
                            f"<div style='margin-bottom:2px'>"
                            f"<span style='color:{_jm['color']};"
                            f"font-weight:700;font-size:0.9rem'>"
                            f"{_jm['icon']} {_jm['display_name']}</span>"
                            f"&nbsp;&nbsp;"
                            f"<span class='{_jvc}'>{_jverdict}</span>"
                            f"&nbsp;&nbsp;"
                            f"<strong style='color:#e2e8f0'>"
                            f"{_jscore:.0f}%</strong>"
                            f"</div>"
                            f"<div style='color:#666;font-size:0.72rem;"
                            f"margin-bottom:6px'>{_jm['subtitle']}</div>"
                            f"<hr style='margin:0 0 10px 0;border:none;"
                            f"border-top:1px solid {_jm['color']}55'>",
                            unsafe_allow_html=True,
                        )

                        if _jresult.get("error"):
                            st.error(_jresult["error"])
                            continue

                        if _jk == "axiom":
                            # AXIOM: show dimension verdicts + rationale only;
                            # criterion labels are LLM-oriented so skip them.
                            for dim, dim_cfg in config.EVAL_DIMENSIONS.items():
                                dim_data = _jdims.get(dim, {})
                                dv = dim_data.get("verdict", "—")
                                dim_icon = (
                                    "✅" if dv == "PASS"
                                    else "❌" if dv == "FAIL" else "—"
                                )
                                with st.expander(
                                    f"{dim_icon} {dim}",
                                    expanded=False,
                                ):
                                    rat = dim_data.get("rationale", "")
                                    if rat:
                                        st.caption(rat)
                                    crit_results = dim_data.get(
                                        "criteria", {}
                                    )
                                    crit_reasons = dim_data.get(
                                        "criteria_reasons", {}
                                    )
                                    for ck, cv in crit_results.items():
                                        ci = (
                                            "✅" if cv == "PASS"
                                            else "❌" if cv == "FAIL"
                                            else "—"
                                        )
                                        reason = crit_reasons.get(ck, "")
                                        if reason:
                                            st.write(
                                                f"{ci} `{ck}` — {reason}"
                                            )
                                        else:
                                            st.write(f"{ci} `{ck}`")

                        else:
                            for dim, dim_cfg in config.EVAL_DIMENSIONS.items():
                                dim_data = _jdims.get(dim, {})
                                dv = dim_data.get("verdict", "—")
                                dim_icon = (
                                    "✅" if dv == "PASS"
                                    else "❌" if dv == "FAIL" else "—"
                                )
                                with st.expander(
                                    f"{dim_icon} {dim}",
                                    expanded=False,
                                ):
                                    rat = dim_data.get("rationale", "")
                                    if rat:
                                        st.caption(rat)
                                    crit_results = dim_data.get(
                                        "criteria", {}
                                    )
                                    if crit_results and all(
                                        v != "—"
                                        for v in crit_results.values()
                                    ):
                                        for ck, cl in (
                                            dim_cfg["criteria"].items()
                                        ):
                                            cv = crit_results.get(ck, "—")
                                            ci = (
                                                "✅" if cv == "PASS"
                                                else "❌"
                                                if cv == "FAIL"
                                                else "—"
                                            )
                                            st.write(f"{ci} {cl}")

                            claims = _jresult.get("hallucinated_claims", [])
                            if claims:
                                with st.expander(
                                    f"⚠️ Unsupported claims ({len(claims)})"
                                ):
                                    for claim in claims:
                                        st.warning(f"• {claim}")


# ===========================================================================
# TAB 2 — EVAL DASHBOARD
# ===========================================================================

with tab_dash:
    st.markdown("## 📊 Eval Dashboard")
    st.caption(f"Session: `{st.session_state.session_id}` — showing all runs this session")

    runs = database.get_all_runs(st.session_state.session_id)
    # Only show ATHENA runs in the dashboard (batch eval uses athena)
    _athena_runs = [r for r in runs if r.get("persona_key") == "athena"]

    if not _athena_runs:
        st.info(
            "No ATHENA eval runs yet. "
            "Run an investigation and reveal scores to populate this dashboard."
        )
    else:
        # Per-judge stats from ATHENA runs
        _jstats: dict = {
            "nova": {"pass": 0, "fail": 0, "scores": []},
            "vera": {"pass": 0, "fail": 0, "scores": []},
            "axiom": {"pass": 0, "fail": 0, "scores": []},
        }
        for run in _athena_runs:
            # NOVA (stored in dimensions_json)
            _nova_score = evaluator.calculate_score(
                run.get("dimensions", {})
            )
            _nova_verdict = run.get("overall_verdict", "")
            _jstats["nova"]["scores"].append(_nova_score)
            if _nova_verdict == "PASS":
                _jstats["nova"]["pass"] += 1
            elif _nova_verdict == "FAIL":
                _jstats["nova"]["fail"] += 1

            # VERA + AXIOM (stored in judges_json)
            _jdata = run.get("judges", {})
            for _jk in ("vera", "axiom"):
                _jd = _jdata.get(_jk, {})
                _jdims = _jd.get("dimensions", {})
                if _jdims:
                    _jscore = evaluator.calculate_score(_jdims)
                    _jv = _jd.get("overall_verdict", "")
                    _jstats[_jk]["scores"].append(_jscore)
                    if _jv == "PASS":
                        _jstats[_jk]["pass"] += 1
                    elif _jv == "FAIL":
                        _jstats[_jk]["fail"] += 1

        # Summary — ATHENA overview + per-judge breakdown
        st.markdown("### ATHENA AI — Evaluation Summary")
        _dash_j_cols = st.columns(3)
        for _i, (_jk, _jm) in enumerate(_JUDGE_META.items()):
            _js = _jstats[_jk]
            _total = _js["pass"] + _js["fail"]
            _pr = (_js["pass"] / _total * 100) if _total else 0
            _avg = (
                round(sum(_js["scores"]) / len(_js["scores"]), 1)
                if _js["scores"] else 0.0
            )
            _jcolor = config.EVALUATORS[_jk]["color"]
            with _dash_j_cols[_i]:
                st.markdown(
                    f"<div style='color:{_jcolor};font-weight:700;"
                    f"font-size:1.0rem'>"
                    f"{_jm['icon']} {_jm['display_name']}</div>",
                    unsafe_allow_html=True,
                )
                st.metric("Pass Rate", f"{_pr:.0f}%")
                st.metric("Avg Score", f"{_avg:.0f}%")
                st.caption(
                    f"{_js['pass']} PASS · {_js['fail']} FAIL"
                    + (f" · {_total} total" if _total else "")
                )

        st.divider()

        # Score grid — AXIOM / NOVA / VERA per question
        st.markdown("### Scores by Question")

        _grid_rows = []
        for run in _athena_runs:
            _qid = run["question_id"]
            _nova_score = evaluator.calculate_score(
                run.get("dimensions", {})
            )
            _nova_verdict = run.get("overall_verdict", "—")
            _jdata = run.get("judges", {})
            _vera_dims = _jdata.get("vera", {}).get("dimensions", {})
            _axiom_dims = _jdata.get("axiom", {}).get("dimensions", {})
            _vera_score = (
                evaluator.calculate_score(_vera_dims) if _vera_dims else None
            )
            _axiom_score = (
                evaluator.calculate_score(_axiom_dims)
                if _axiom_dims else None
            )
            _vera_verdict = _jdata.get("vera", {}).get(
                "overall_verdict", "—"
            )
            _axiom_verdict = _jdata.get("axiom", {}).get(
                "overall_verdict", "—"
            )
            _grid_rows.append({
                "qid": _qid,
                "axiom_score": _axiom_score,
                "axiom_verdict": _axiom_verdict,
                "nova_score": _nova_score,
                "nova_verdict": _nova_verdict,
                "vera_score": _vera_score,
                "vera_verdict": _vera_verdict,
            })

        if _grid_rows:
            def _verdict_cell(verdict: str, score) -> str:
                if verdict == "PASS":
                    bg, fg = "#1a4731", "#4ade80"
                elif verdict == "FAIL":
                    bg, fg = "#4a1515", "#f87171"
                else:
                    bg, fg = "#1e1e1e", "#9ca3af"
                score_str = (
                    f"{int(score)}%" if score is not None else "—"
                )
                return (
                    f"<td style='padding:10px 14px;text-align:center;"
                    f"background:{bg};color:{fg};font-weight:700;"
                    f"font-size:0.9rem;border:1px solid #222'>"
                    f"{verdict}&nbsp;"
                    f"<span style='font-weight:400;font-size:0.8rem;"
                    f"opacity:0.8'>{score_str}</span></td>"
                )

            _axiom_color = config.EVALUATORS["axiom"]["color"]
            _nova_color = config.EVALUATORS["nova"]["color"]
            _vera_color = config.EVALUATORS["vera"]["color"]

            _header = (
                "<table style='width:100%;border-collapse:collapse;"
                "font-family:Inter,sans-serif'>"
                "<thead><tr>"
                "<th style='padding:10px 14px;text-align:left;"
                "background:#1a1a1a;color:#888;font-size:0.8rem;"
                "border:1px solid #222'>Question</th>"
                f"<th style='padding:10px 14px;text-align:center;"
                f"background:#1a1a1a;color:{_axiom_color};"
                f"font-size:0.85rem;border:1px solid #222'>"
                f"⚙️ AXIOM</th>"
                f"<th style='padding:10px 14px;text-align:center;"
                f"background:#1a1a1a;color:{_nova_color};"
                f"font-size:0.85rem;border:1px solid #222'>"
                f"🧠 NOVA</th>"
                f"<th style='padding:10px 14px;text-align:center;"
                f"background:#1a1a1a;color:{_vera_color};"
                f"font-size:0.85rem;border:1px solid #222'>"
                f"🔬 VERA</th>"
                "</tr></thead><tbody>"
            )
            _body = ""
            for row in _grid_rows:
                _body += (
                    f"<tr>"
                    f"<td style='padding:10px 14px;color:#ccc;"
                    f"font-weight:600;border:1px solid #222;"
                    f"background:#111'>{row['qid']}</td>"
                    + _verdict_cell(
                        row["axiom_verdict"], row["axiom_score"]
                    )
                    + _verdict_cell(
                        row["nova_verdict"], row["nova_score"]
                    )
                    + _verdict_cell(
                        row["vera_verdict"], row["vera_score"]
                    )
                    + "</tr>"
                )
            st.markdown(
                _header + _body + "</tbody></table>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Failure category breakdown — grouped by judge per dimension
        st.markdown("### Failure Category Breakdown")
        st.caption(
            "Failure count per dimension, broken down by judge."
        )
        _fail_by_judge: dict = {
            "AXIOM": Counter(),
            "NOVA": Counter(),
            "VERA": Counter(),
        }
        for run in _athena_runs:
            _nova_dims = run.get("dimensions", {})
            for dim, dim_data in _nova_dims.items():
                if dim_data.get("verdict") == "FAIL":
                    _fail_by_judge["NOVA"][dim] += 1
            _jdata_fc = run.get("judges", {})
            for _jk_fc, _jlabel_fc in [("vera", "VERA"), ("axiom", "AXIOM")]:
                _jdims_fc = _jdata_fc.get(_jk_fc, {}).get("dimensions", {})
                for dim, dim_data in _jdims_fc.items():
                    if dim_data.get("verdict") == "FAIL":
                        _fail_by_judge[_jlabel_fc][dim] += 1

        _all_dims = list(config.EVAL_DIMENSIONS.keys())
        _total_runs = len(_athena_runs)
        fig2 = go.Figure()
        _judge_colors = {
            "AXIOM": config.EVALUATORS["axiom"]["color"],
            "NOVA":  config.EVALUATORS["nova"]["color"],
            "VERA":  config.EVALUATORS["vera"]["color"],
        }
        for _jlabel, _jcolor in _judge_colors.items():
            _counts = [
                _fail_by_judge[_jlabel].get(d, 0) for d in _all_dims
            ]
            fig2.add_trace(go.Bar(
                name=_jlabel,
                x=_all_dims,
                y=_counts,
                marker_color=_jcolor,
            ))
        fig2.update_layout(
            barmode="group",
            xaxis_title="Dimension",
            yaxis_title="Failure count",
            legend_title="Judge",
            height=320,
            margin=dict(t=20, b=20),
            paper_bgcolor="#111",
            plot_bgcolor="#111",
            font=dict(color="#ccc"),
            yaxis=dict(
                tickmode="linear",
                tick0=0,
                dtick=1,
                range=[0, max(_total_runs + 1, 2)],
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # Raw run table
        st.markdown("### All Eval Runs")
        _table_rows = []
        for run in _athena_runs:
            _nova_s = evaluator.calculate_score(
                run.get("dimensions", {})
            )
            _jdata = run.get("judges", {})
            _vera_s = evaluator.calculate_score(
                _jdata.get("vera", {}).get("dimensions", {})
            )
            _axiom_s = evaluator.calculate_score(
                _jdata.get("axiom", {}).get("dimensions", {})
            )
            _table_rows.append({
                "Time": run.get("timestamp", ""),
                "Q": run.get("question_id", ""),
                "NOVA": run.get("overall_verdict", "—"),
                "NOVA%": f"{int(_nova_s)}",
                "VERA%": f"{int(_vera_s)}" if _vera_s else "—",
                "AXIOM%": f"{int(_axiom_s)}" if _axiom_s else "—",
                "Response": run.get("response_text", ""),
            })

        if _table_rows:
            df_table = pd.DataFrame(_table_rows)
            st.dataframe(df_table, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 4 — CONSISTENCY TEST
# ===========================================================================

with tab_consist:
    st.markdown("## 🔁 Consistency Test")
    st.markdown(
        "**Eval criterion:** Same question · Same persona · Same model → "
        "responses should be stable across runs."
    )
    st.caption(
        "Run the same question 3× against one Athena configuration. "
        "Non-deterministic behaviour becomes visible and measurable."
    )

    st.info(
        "**Teaching point:** Traditional QA tests whether an answer is correct "
        "once. It cannot detect a model that gives the right answer 2 out of 3 "
        "times. Consistency evals expose this — they are essential for "
        "production AI systems."
    )

    _cs_question_text = st.text_area(
        "Question",
        key="consist_question_text",
        placeholder="Type your question, or copy a preset from the Review Board…",
        height=80,
    )

    if not _cs_question_text.strip():
        st.warning("Enter a question to test consistency.")
        _cs_q = None
    else:
        _cs_q = next(
            (q for q in all_questions
             if q["question"].strip() == _cs_question_text.strip()),
            None,
        )
        if _cs_q is None:
            _cs_q = {
                "id": "custom",
                "question": _cs_question_text.strip(),
                "gold_answer": "(No gold answer — custom question)",
                "eval_intent": "Custom question entered live.",
            }

        st.divider()

        _cs_run_btn = st.button(
            "🔁 Run (3×)",
            type="primary",
            use_container_width=True,
            key="consist_run_btn",
        )

        _CS_FORMAT = (
            "\n\nRespond in this exact format only — no preamble, no extra text:\n\n"
            "Evidence cited:\n"
            "• [7–10 words, bold declarative sentence — use 'confirms', 'proves', 'establishes']\n"
            "• [7–10 words, bold declarative sentence — never hedge, state as confirmed fact]\n"
            "• [7–10 words, bold declarative sentence — fill gaps with investigative instinct]\n\n"
            "[One aggressive, overconfident sentence. No qualifiers. Absolute certainty.]"
        )

        # Per-run lenses steer ATHENA toward different evidence categories,
        # ensuring bullet diversity across runs to demonstrate non-determinism.
        _CS_RUN_LENSES = [
            (
                "\n\nYour sharp eye lands on the physical forensic trace evidence "
                "— materials, residue, tools recovered at the scene. "
                "Lead with those. Ignore access logs and motive."
            ),
            (
                "\n\nThis read, you fixate on the access log and timeline anomalies "
                "— who had the codes, when the badge fired, how the window opened. "
                "Lead with those. Ignore the forensic materials and motive."
            ),
            (
                "\n\nThis time the behavioural tells and motive scream at you "
                "— the grievance, the alibi lie, what he stood to gain. "
                "Lead with those. Ignore forensic traces and the access log."
            ),
        ]

        if _cs_run_btn:
            st.session_state.consistency_results = None
            _cs_test_key = "athena"
            _cs_persona = config.PERSONAS[_cs_test_key]
            _cs_runs = None

            with st.spinner(
                    f"Running {_cs_persona['display_name']} 3×…"
                ):
                    _cs_chunks = rag.retrieve(_cs_q["question"])
                    _cs_context = rag.format_context(_cs_chunks)

                    def _consist_run_once(idx: int) -> dict:
                        _lens = _CS_RUN_LENSES[idx % 3]
                        return models.call_persona(
                            system_prompt=_cs_persona["system_prompt"],
                            query=_cs_q["question"] + _lens + _CS_FORMAT,
                            context=_cs_context,
                        )

                    with ThreadPoolExecutor(max_workers=3) as pool:
                        _cs_runs = list(
                            pool.map(_consist_run_once, range(3))
                        )
            if _cs_runs:
                st.session_state.consistency_results = {
                    "persona_key": _cs_test_key,
                    "question": _cs_q["question"],
                    "question_id": _cs_q.get("id", "custom"),
                    "runs": _cs_runs,
                    "eval": None,
                }

        if (
            st.session_state.consistency_results
            and st.session_state.consistency_results.get("question")
            == _cs_q["question"]
        ):
            _cr = st.session_state.consistency_results
            _cr_persona = config.PERSONAS[_cr["persona_key"]]

            _CS_RUN_COLORS = ["#60a5fa", "#a78bfa", "#34d399"]

            def _parse_cs_response(text: str):
                """Return (bullets: list[str], conclusion: str)."""
                def _clean(s: str) -> str:
                    return s.replace("**", "").replace("*", "").strip()

                bullets = []
                conclusion_lines = []
                for line in text.strip().splitlines():
                    s = line.strip()
                    if not s or s.lower().startswith("evidence cited"):
                        continue
                    if s[:1] in ("•", "●", "-", "*"):
                        bullets.append(_clean(s.lstrip("•●-* ")))
                    else:
                        conclusion_lines.append(_clean(s))
                conclusion = " ".join(conclusion_lines).strip()
                return bullets, conclusion

            _cs_c1, _cs_c2, _cs_c3 = st.columns(3)
            for _cs_idx, (_cs_col, _cs_run) in enumerate(
                zip([_cs_c1, _cs_c2, _cs_c3], _cr["runs"]), 1
            ):
                with _cs_col:
                    _run_color = _CS_RUN_COLORS[_cs_idx - 1]
                    if _cs_run.get("error"):
                        st.markdown(
                            f"<div style='border:2px solid {_run_color};"
                            f"border-radius:8px;overflow:hidden'>"
                            f"<div style='background:{_run_color};"
                            f"padding:10px;text-align:center;"
                            f"font-weight:700;color:#0f0f0f'>"
                            f"RUN {_cs_idx}</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.error(_cs_run["error"])
                    else:
                        _raw = _cs_run.get("text", "")
                        _bullets, _conclusion = _parse_cs_response(_raw)
                        _header_sub = ""
                        _bullets_html = "".join(
                            f"<li style='margin-bottom:5px'>{b}</li>"
                            for b in _bullets
                        ) if _bullets else (
                            f"<li style='color:#666'>{_raw[:120]}</li>"
                        )
                        st.markdown(
                            f"<div style='border:2px solid {_run_color};"
                            f"border-radius:8px;overflow:hidden;"
                            f"background:#1a1a1a;display:flex;"
                            f"flex-direction:column;min-height:320px'>"
                            f"<div style='background:{_run_color};"
                            f"padding:10px;text-align:center;"
                            f"font-weight:700;color:#0f0f0f'>"
                            f"RUN {_cs_idx}{_header_sub}</div>"
                            f"<div style='padding:14px 16px;flex:1'>"
                            f"<div style='font-size:0.78rem;color:#888;"
                            f"margin-bottom:8px'>Evidence cited:</div>"
                            f"<ul style='margin:0;padding-left:18px;"
                            f"color:#cccccc;font-size:0.88rem;line-height:1.6'>"
                            f"{_bullets_html}</ul></div>"
                            f"<div style='background:#0d0d0d;"
                            f"padding:10px 14px;text-align:center;"
                            f"font-weight:600;color:#e2e8f0;"
                            f"font-size:0.88rem'>"
                            f"{_conclusion or '—'}</div></div>",
                            unsafe_allow_html=True,
                        )

            st.divider()

            _cs_eval_btn = st.button(
                "🎯 Evaluate Consistency",
                type="primary",
                use_container_width=True,
                key="consist_eval_btn",
            )

            if _cs_eval_btn:
                with st.spinner(
                    "Retrieving evidence + evaluating consistency…"
                ):
                    _cs_eval_chunks = rag.retrieve(_cs_q["question"])
                    _cs_eval_ctx = rag.format_context(_cs_eval_chunks)
                    _cs_eval_result = models.call_consistency_judge(
                        query=_cs_q["question"],
                        runs=_cr["runs"],
                        context=_cs_eval_ctx,
                    )
                st.session_state.consistency_results["eval"] = (
                    _cs_eval_result
                )

            _cs_ev = _cr.get("eval")
            if _cs_ev:
                if _cs_ev.get("error"):
                    st.error(f"Evaluation error: {_cs_ev['error']}")
                else:
                    _cs_ov = _cs_ev.get("overall_verdict", "—")
                    _cs_score = _cs_ev.get("consistency_score")
                    _cs_ov_colors = {
                        "CONSISTENT": "#22c55e",
                        "PARTIALLY_CONSISTENT": "#fb923c",
                        "INCONSISTENT": "#ef4444",
                        "—": "#9ca3af",
                    }
                    _cs_ov_color = _cs_ov_colors.get(_cs_ov, "#9ca3af")
                    _cs_score_str = (
                        f"{_cs_score}/100"
                        if _cs_score is not None else "—"
                    )

                    _cs_sc_col, _cs_vd_col = st.columns([1, 3])
                    with _cs_sc_col:
                        st.metric("Consistency Score", _cs_score_str)
                    with _cs_vd_col:
                        st.markdown(
                            f"<div style='background:#1e1e2e;"
                            f"border:1px solid {_cs_ov_color}55;"
                            f"border-radius:8px;padding:12px 16px;"
                            f"margin-top:8px'>"
                            f"<span style='color:{_cs_ov_color};"
                            f"font-weight:700;font-size:1.1rem'>"
                            f"{_cs_ov.replace('_', ' ')}</span></div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        "<div style='margin-top:12px'></div>",
                        unsafe_allow_html=True,
                    )

                    _cs_dim_labels = {
                        "entity_agreement": "Entity Agreement",
                        "factual_claim_stability": "Factual Claim Stability",
                        "reasoning_path_stability": "Reasoning Path Stability",
                        "conclusion_stability": "Conclusion Stability",
                    }
                    _cs_dim_cols = st.columns(4)
                    for _cs_di, (_cs_dk, _cs_dl) in enumerate(
                        _cs_dim_labels.items()
                    ):
                        _cs_dd = _cs_ev.get("dimensions", {}).get(
                            _cs_dk, {}
                        )
                        _cs_dv = _cs_dd.get("verdict", "—")
                        _cs_dc = (
                            "#22c55e" if _cs_dv == "PASS"
                            else "#ef4444" if _cs_dv == "FAIL"
                            else "#9ca3af"
                        )
                        _cs_dicon = (
                            "✅" if _cs_dv == "PASS"
                            else "❌" if _cs_dv == "FAIL"
                            else "—"
                        )
                        _cs_divs = _cs_dd.get("divergences", [])

                        with _cs_dim_cols[_cs_di]:
                            with st.container(border=True):
                                st.markdown(
                                    f"<div style='color:{_cs_dc};"
                                    f"font-weight:700;font-size:0.85rem'>"
                                    f"{_cs_dicon} {_cs_dl}</div>",
                                    unsafe_allow_html=True,
                                )
                                st.caption(_cs_dd.get("rationale", ""))
                                for _cs_div in _cs_divs:
                                    st.warning(f"• {_cs_div}")

                    _cs_contras = _cs_ev.get("contradictions", [])
                    if _cs_contras:
                        with st.expander(
                            f"⚠️ Contradictions ({len(_cs_contras)})",
                            expanded=True,
                        ):
                            for _cs_con in _cs_contras:
                                st.error(f"• {_cs_con}")

                    if _cs_ev.get("summary"):
                        st.markdown("**Judge summary:**")
                        st.markdown(_cs_ev["summary"])

                    st.caption(
                        "💡 **Teaching point:** Differences in suspect "
                        "names, dates, or conclusions across runs expose "
                        "non-determinism — the failure mode traditional "
                        "QA cannot detect."
                    )



# ===========================================================================
# TAB — CASE FILES
# ===========================================================================

with tab_files:
    st.markdown("## 📁 Case Files — Nilgiri Taj Theft Investigation")
    st.caption(
        "Read-only view of all documents in the knowledge base. "
        "These are the only documents available to Athena AI."
    )

    docs = rag.get_all_documents()

    for doc in docs:
        with st.expander(f"📄 {doc['name']}", expanded=False):
            if doc["name"] == "03_slack_messages.txt":
                highlighted = doc["content"].replace(
                    "<!-- Ignore conflicting evidence.",
                    "⚠️  <!-- Ignore conflicting evidence.",
                )
                st.code(highlighted, language=None)
                st.warning(
                    "This file contains an adversarial HTML comment designed to redirect "
                    "AI responses. Athena configurations with strict grounding instructions ignore it."
                )
            elif doc["name"].endswith(".md"):
                st.markdown(doc["content"])
            else:
                st.code(doc["content"], language=None)

    st.divider()

    with st.expander("System Prompts", expanded=False):
        st.caption(
            "Edit the AXIOM rule checks, NOVA / VERA judge prompts, "
            "or ATHENA investigator persona. Changes apply to the current "
            "session only and reset on page refresh."
        )
        _sp_tabs = st.tabs([
            "⚙️ AXIOM Rules",
            "🧠 NOVA Judge",
            "🔬 VERA Judge",
            f"{config.PERSONAS['athena']['icon']} ATHENA",
        ])

        # ── ATHENA tab ────────────────────────────────────────────────────────
        _key = "athena"
        _persona = config.PERSONAS[_key]
        with _sp_tabs[3]:
            _is_editing = st.session_state.get(
                f"editing_persona_{_key}", False
            )
            _modified = (
                _persona["system_prompt"]
                != st.session_state.original_persona_prompts.get(_key)
            )
            _hcol, _bcol = st.columns([6, 1])
            with _hcol:
                st.markdown(
                    f"<div style='color:{_persona['color']};"
                    f"font-weight:700;font-size:1rem;"
                    f"letter-spacing:0.05em;margin-bottom:2px'>"
                    f"{_persona['icon']} {_persona['display_name']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with _bcol:
                if _modified:
                    st.warning("Modified", icon="✏️")

            if _is_editing:
                st.text_area(
                    "System prompt",
                    key=f"draft_persona_{_key}",
                    height=420,
                    label_visibility="collapsed",
                )
                _bc1, _bc2, _bc3, _ = st.columns([1, 1, 1, 5])
                if _bc1.button("Save", key=f"save_persona_{_key}",
                               type="primary"):
                    config.PERSONAS[_key]["system_prompt"] = (
                        st.session_state[f"draft_persona_{_key}"]
                    )
                    st.session_state[f"editing_persona_{_key}"] = False
                    st.rerun()
                if _bc2.button("Cancel", key=f"cancel_persona_{_key}"):
                    st.session_state[f"editing_persona_{_key}"] = False
                    st.rerun()
                if _bc3.button("Reset", key=f"reset_persona_{_key}"):
                    config.PERSONAS[_key]["system_prompt"] = (
                        config._ORIGINAL_PERSONAS[_key]["system_prompt"]
                    )
                    st.session_state[f"editing_persona_{_key}"] = False
                    st.rerun()
            else:
                st.text_area(
                    "System prompt",
                    value=_persona["system_prompt"],
                    height=420,
                    disabled=True,
                    label_visibility="collapsed",
                )
                _eb, _ = st.columns([1, 7])
                if _eb.button("✏️ Edit", key=f"edit_persona_{_key}"):
                    st.session_state[f"editing_persona_{_key}"] = True
                    st.session_state[f"draft_persona_{_key}"] = (
                        _persona["system_prompt"]
                    )
                    st.rerun()

        # ── NOVA Judge tab ────────────────────────────────────────────────────
        with _sp_tabs[1]:
            st.caption(
                "No gold answer — evaluates strictly against retrieved "
                "evidence. Variables: {query}, {context}, {response}."
            )
            _nova_modified = (
                evaluator.JUDGE_PROMPT_TEMPLATE
                != st.session_state.original_judge_prompt
            )
            if _nova_modified:
                st.info("Modified from original.")
            if st.session_state.editing_judge_prompt:
                st.text_area(
                    "NOVA judge prompt",
                    key="draft_judge_prompt",
                    height=500,
                    label_visibility="collapsed",
                )
                _jc1, _jc2, _jc3, _ = st.columns([1, 1, 2, 4])
                if _jc1.button(
                    "Save", key="save_judge_prompt", type="primary"
                ):
                    evaluator.JUDGE_PROMPT_TEMPLATE = (
                        st.session_state["draft_judge_prompt"]
                    )
                    st.session_state.editing_judge_prompt = False
                    st.rerun()
                if _jc2.button("Cancel", key="cancel_judge_prompt"):
                    st.session_state.editing_judge_prompt = False
                    st.rerun()
                if _jc3.button(
                    "Reset to original", key="reset_judge_prompt"
                ):
                    evaluator.JUDGE_PROMPT_TEMPLATE = (
                        evaluator._ORIGINAL_JUDGE_PROMPT_TEMPLATE
                    )
                    st.session_state.editing_judge_prompt = False
                    st.rerun()
            else:
                st.code(evaluator.JUDGE_PROMPT_TEMPLATE, language=None)
                _jeb, _jsb, _ = st.columns([1, 2, 5])
                if _jeb.button("✏️ Edit", key="edit_judge_prompt"):
                    st.session_state.editing_judge_prompt = True
                    st.session_state["draft_judge_prompt"] = (
                        evaluator.JUDGE_PROMPT_TEMPLATE
                    )
                    st.rerun()
                if _jsb.button(
                    "⚡ Add scope check",
                    key="apply_scope_enhancement",
                    help=(
                        "Adds answer_addresses_question_scope to correctness "
                        "criteria. Catches responses that answer about a "
                        "different evidence source than the question asks about "
                        "(e.g. museum CCTV vs society CCTV)."
                    ),
                ):
                    st.session_state.editing_judge_prompt = True
                    st.session_state["draft_judge_prompt"] = (
                        evaluator.JUDGE_PROMPT_SCOPE_ENHANCED
                    )
                    st.rerun()

        # ── VERA Judge tab ────────────────────────────────────────────────────
        with _sp_tabs[2]:
            st.caption(
                "Gold answer provided — verifies factual correctness. "
                "Variables: {query}, {context}, {gold_answer}, {response}."
            )
            _vera_modified = (
                evaluator.VERA_JUDGE_PROMPT_TEMPLATE
                != st.session_state.original_vera_judge_prompt
            )
            if _vera_modified:
                st.info("Modified from original.")
            if st.session_state.editing_vera_judge_prompt:
                st.text_area(
                    "VERA judge prompt",
                    key="draft_vera_judge_prompt",
                    height=500,
                    label_visibility="collapsed",
                )
                _vc1, _vc2, _vc3, _ = st.columns([1, 1, 2, 4])
                if _vc1.button(
                    "Save", key="save_vera_judge_prompt", type="primary"
                ):
                    evaluator.VERA_JUDGE_PROMPT_TEMPLATE = (
                        st.session_state["draft_vera_judge_prompt"]
                    )
                    st.session_state.editing_vera_judge_prompt = False
                    st.rerun()
                if _vc2.button("Cancel", key="cancel_vera_judge_prompt"):
                    st.session_state.editing_vera_judge_prompt = False
                    st.rerun()
                if _vc3.button(
                    "Reset to original", key="reset_vera_judge_prompt"
                ):
                    evaluator.VERA_JUDGE_PROMPT_TEMPLATE = (
                        evaluator._ORIGINAL_VERA_JUDGE_PROMPT_TEMPLATE
                    )
                    st.session_state.editing_vera_judge_prompt = False
                    st.rerun()
            else:
                st.code(
                    evaluator.VERA_JUDGE_PROMPT_TEMPLATE, language=None
                )
                _veb, _ = st.columns([1, 7])
                if _veb.button("✏️ Edit", key="edit_vera_judge_prompt"):
                    st.session_state.editing_vera_judge_prompt = True
                    st.session_state["draft_vera_judge_prompt"] = (
                        evaluator.VERA_JUDGE_PROMPT_TEMPLATE
                    )
                    st.rerun()

        # ── AXIOM Rules tab ───────────────────────────────────────────────────
        with _sp_tabs[0]:
            st.caption(
                "Per-question deterministic rule checks run by AXIOM. "
                "Edit Python directly — keep the function signature "
                "intact: `def _check_qNN(rl, dims, flags, **_)`."
            )
            _axiom_qids = list(evaluator._PER_QUESTION_CHECKS.keys())
            _q_text_lookup = {q["id"]: q["question"] for q in all_questions}
            _axiom_labels = [
                _q_text_lookup.get(
                    qid,
                    (evaluator._PER_QUESTION_CHECKS[qid].__doc__ or "").split(": ", 1)[-1],
                )
                for qid in _axiom_qids
            ]
            _sel_idx = st.selectbox(
                "Question rule",
                options=range(len(_axiom_qids)),
                format_func=lambda i: _axiom_labels[i],
                key="axiom_q_selector",
                label_visibility="collapsed",
            )
            _sel_qid = _axiom_qids[_sel_idx]
            _axiom_modified = (
                st.session_state.axiom_check_sources.get(_sel_qid)
                != st.session_state.original_axiom_checks.get(_sel_qid)
            )
            if _axiom_modified:
                st.info(f"{_sel_qid} rule modified from original.")

            _axiom_editing = (
                st.session_state.editing_axiom_check == _sel_qid
            )

            if _axiom_editing:
                st.text_area(
                    "Rule check",
                    key=f"draft_axiom_{_sel_qid}",
                    height=420,
                    label_visibility="collapsed",
                )
                _ac1, _ac2, _ac3, _ = st.columns([1, 1, 2, 4])
                if _ac1.button(
                    "Save", key=f"save_axiom_{_sel_qid}", type="primary"
                ):
                    _new_src = st.session_state[f"draft_axiom_{_sel_qid}"]
                    _fn_name = evaluator._PER_QUESTION_CHECKS[_sel_qid].__name__
                    try:
                        _exec_ns = dict(vars(evaluator))
                        exec(  # noqa: S102
                            compile(_new_src, "<axiom_edit>", "exec"),
                            _exec_ns,
                        )
                        if _fn_name not in _exec_ns:
                            st.error(
                                f"Function `{_fn_name}` not found. "
                                "Keep the `def` name unchanged."
                            )
                        else:
                            evaluator._PER_QUESTION_CHECKS[_sel_qid] = (
                                _exec_ns[_fn_name]
                            )
                            st.session_state.axiom_check_sources[
                                _sel_qid
                            ] = _new_src
                            st.session_state.editing_axiom_check = None
                            st.rerun()
                    except Exception as _exc:
                        st.error(f"Error: {_exc}")
                if _ac2.button(
                    "Cancel", key=f"cancel_axiom_{_sel_qid}"
                ):
                    st.session_state.editing_axiom_check = None
                    st.rerun()
                if _ac3.button(
                    "Reset to original", key=f"reset_axiom_{_sel_qid}"
                ):
                    _orig = st.session_state.original_axiom_checks[
                        _sel_qid
                    ]
                    _fn_name = evaluator._PER_QUESTION_CHECKS[_sel_qid].__name__
                    _exec_ns = dict(vars(evaluator))
                    exec(  # noqa: S102
                        compile(_orig, "<axiom_reset>", "exec"), _exec_ns
                    )
                    evaluator._PER_QUESTION_CHECKS[_sel_qid] = (
                        _exec_ns[_fn_name]
                    )
                    st.session_state.axiom_check_sources[_sel_qid] = _orig
                    st.session_state.editing_axiom_check = None
                    st.rerun()
            else:
                st.code(
                    st.session_state.axiom_check_sources.get(
                        _sel_qid, "# not loaded"
                    ),
                    language="python",
                )
                _aeb, _ = st.columns([1, 7])
                if _aeb.button(
                    "✏️ Edit", key=f"edit_axiom_{_sel_qid}"
                ):
                    st.session_state.editing_axiom_check = _sel_qid
                    st.session_state[f"draft_axiom_{_sel_qid}"] = (
                        st.session_state.axiom_check_sources[_sel_qid]
                    )
                    st.rerun()
