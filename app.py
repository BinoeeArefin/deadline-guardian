"""
Deadline Guardian - GDG on Campus Debrecen, Agentic AI Hackathon 2026.

Paste a syllabus / Neptun message / teacher email (English or Hungarian).
Gemini extracts the deadlines and estimates how many days of work each one
needs. Fixed, explainable rules (planner.py) then warn you early enough to
actually finish.

Run:  streamlit run app.py      (needs GEMINI_API_KEY)
"""

import html
import json
import os
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

from planner import (
    STATUS_AT_RISK, STATUS_ON_TRACK, STATUS_START_TODAY,
    advice, build_plan, detect_overload, explain_missed_day, fmt_day,
    snapshot, start_by_date, task_status,
)

PRIMARY = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
FALLBACKS = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
             "gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-flash-lite-latest",
             "gemini-3.8-flash"]
MODELS = [PRIMARY] + [m for m in FALLBACKS if m != PRIMARY]
MODEL_LABEL = f"Gemini (auto: tries {MODELS[0]} first, falls back if busy)"
RETRYABLE_CODES = {404, 429, 500, 503, 504}  # model gone / rate-limited / server error / overloaded / deadline exceeded
TZ = ZoneInfo("Europe/Budapest")
TYPES = ["exam", "assignment", "project", "presentation"]
TYPE_ICON = {"exam": "📝", "assignment": "📄", "project": "🧩", "presentation": "🎤"}
SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLES = {
    "🇬🇧 English syllabus": SAMPLES_DIR / "syllabus_en.txt",
    "🇭🇺 Hungarian Neptun message": SAMPLES_DIR / "neptun_hu.txt",
}

st.set_page_config(page_title="Deadline Guardian", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "tasks": [],              # list of task dicts (source of truth)
        "next_id": 1,
        "editor_version": 0,      # bump to reset the data editor
        "missed_days": 0,
        "missed_explanation": None,
        "input_text": "",
        "flash": None,            # one-off message after extraction
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def real_today() -> date:
    return datetime.now(TZ).date()


def base_today() -> date:
    return st.session_state.get("pretend_today") or real_today()


def planning_today() -> date:
    return base_today() + timedelta(days=st.session_state.missed_days)


init_state()


# ----------------------------------------------------------------------------
# Gemini: extraction + effort estimate (features 1 and 2)
# ----------------------------------------------------------------------------
class ExtractedTask(BaseModel):
    task: str = Field(description="Short name of the graded task in English. If the text is not in English, "
                                  "add the original term in brackets, e.g. 'Second midterm (2. ZH)', "
                                  "'Written exam (Írásbeli vizsga)'.")
    course: str = Field(description="Course / subject name, '' if unknown.")
    type: str = Field(description="Exactly one of: exam, assignment, project, presentation.")
    deadline: str = Field(description="Local Europe/Budapest time as YYYY-MM-DDTHH:MM, or '' if no date is given.")
    location: str = Field(description="Room, building or platform (e.g. IK-F01, Moodle). '' if none.")
    effort_days: int = Field(description="Minimum full days of focused work a typical student needs. Integer >= 1.")
    effort_reason: str = Field(description="One short line in English explaining the estimate, citing cues from the text.")
    evidence: str = Field(description="Short exact quote (max ~15 words) from the text where this deadline appears.")


class Extraction(BaseModel):
    tasks: list[ExtractedTask]


SYSTEM_PROMPT = """You are the extraction step of Deadline Guardian, an assistant for University of Debrecen students.
The user pastes a syllabus, a Neptun message or a teacher email, in English or Hungarian.

Extract EVERY graded item the student must prepare for that has a date:
- exam: exams, midterms, quizzes, ZH / zárthelyi dolgozat, vizsga, kollokvium
- assignment: homework, lab assignment, beadandó, házi feladat, lab report
- project: projects, final reports with code, group projects
- presentation: presentations, prezentáció, projektbemutató, kiselőadás
Ignore lectures, office hours, konzultáció, holidays and administrative notes that need no preparation.
Write `task` in English. For Hungarian (or any non-English) text, keep the original term in brackets after the English name so the student can match it to Neptun/Moodle. `course` stays as written in the text. `evidence` stays as the exact original quote.

Rules for `deadline`:
- Output local Europe/Budapest time as YYYY-MM-DDTHH:MM.
- If no time is given, use 23:59.
- If no year is given, use the next occurrence on or after today.
- Resolve relative dates ("next Friday", "jövő hét kedd") from today's date given below.
- Hungarian formats like "2026. november 24. 23:59" or "2026.11.30. 8:00" are common.
- If the item has no date at all, output "".

Rules for `effort_days` (minimum days of real work, integer, never 0):
- small assignment / short quiz: 1
- larger assignment (many exercises, big weight): 2-3
- midterm / ZH: 2-3
- final / written exam covering the whole semester: 3-5
- presentation: 1-2 (more if the content must also be produced)
- project / report with code: 3-7 depending on size, page count, team size and weight
Use cues in the text (weight %, number of exercises, page count, topics covered, "about 2 hours", team size).
`effort_reason` is one short English line naming the cue you used.

Never invent tasks, dates, rooms or numbers that are not supported by the text.
If the text contains no graded items, return an empty list."""


def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def gemini_client(api_key: str):
    from google import genai
    from google.genai import types
    # 90 s timeout per request and no internal SDK retries, so a busy model can't hang the spinner forever
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=90_000, retry_options=types.HttpRetryOptions(attempts=1)))


def _is_timeout(e: Exception) -> bool:
    try:
        import httpx
        if isinstance(e, httpx.TimeoutException):
            return True
    except ImportError:
        pass
    return isinstance(e, TimeoutError) or "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower()


@st.cache_data(show_spinner=False, ttl=3600)
def call_gemini(text: str, today_iso: str, api_key: str) -> tuple:
    """Real Gemini call with automatic model fallback.
    Returns (tasks, model_used). Cached per (text, date) so re-running a demo is fast.
    Errors are raised (and never cached)."""
    from google.genai import errors, types

    today_d = date.fromisoformat(today_iso)
    prompt = (f"Today is {today_d.strftime('%A')}, {today_iso} (Europe/Budapest).\n\n"
              f"TEXT START\n{text}\nTEXT END")
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=Extraction,
        temperature=0.2,
    )

    failures = []
    for model in MODELS:
        try:
            resp = gemini_client(api_key).models.generate_content(
                model=model, contents=prompt, config=config)
        except errors.APIError as e:
            if getattr(e, "code", None) in RETRYABLE_CODES:
                failures.append(f"{model}: {e.code} {getattr(e, 'status', '') or ''}".strip())
                continue
            raise  # e.g. 400 bad request / 403 invalid key: another model won't help
        except Exception as e:
            if _is_timeout(e):
                failures.append(f"{model}: timeout after 90 s")
                continue
            raise
        if not resp.text:
            raise RuntimeError(f"Gemini ({model}) returned an empty response (it may have been blocked). "
                               "Try shorter text.")
        data = json.loads(resp.text)
        tasks = data.get("tasks", []) if isinstance(data, dict) else data
        return tasks, model

    raise RuntimeError("All Gemini models failed, nothing was extracted:\n"
                       + "\n".join(f"- {f}" for f in failures))


# ----------------------------------------------------------------------------
# Normalisation helpers (shared by AI results and table edits)
# ----------------------------------------------------------------------------
def clean_str(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def clean_type(v) -> str:
    v = clean_str(v).lower()
    return v if v in TYPES else "assignment"


def clean_effort(v) -> int:
    try:
        if v is None or pd.isna(v):
            return 1
        return min(60, max(1, int(round(float(v)))))
    except (TypeError, ValueError):
        return 1


def parse_deadline(s):
    s = clean_str(s)
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = pd.Timestamp(s).to_pydatetime()
        except Exception:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(TZ).replace(tzinfo=None)
    if len(s) == 10:  # date only
        dt = dt.replace(hour=23, minute=59)
    return dt.replace(second=0, microsecond=0)


def new_id() -> int:
    i = st.session_state.next_id
    st.session_state.next_id += 1
    return i


def task_from_ai(raw: dict) -> dict:
    effort = clean_effort(raw.get("effort_days"))
    return {
        "id": new_id(),
        "task": clean_str(raw.get("task")) or "(untitled task)",
        "course": clean_str(raw.get("course")),
        "type": clean_type(raw.get("type")),
        "deadline": parse_deadline(raw.get("deadline")),
        "location": clean_str(raw.get("location")),
        "effort_days": effort,
        "ai_effort_days": effort,
        "effort_reason": clean_str(raw.get("effort_reason")),
        "evidence": clean_str(raw.get("evidence")),
    }


def signature(tasks: list) -> tuple:
    return tuple((t["id"], t["task"], t["course"], t["type"], t["deadline"], t["effort_days"],
                  t["location"], t["effort_reason"]) for t in tasks)


EDITOR_COLS = ["Task", "Course", "Type", "Deadline", "Effort (days)", "AI said", "Why (AI)", "Location", "Evidence"]


def tasks_to_df(tasks: list) -> pd.DataFrame:
    rows = [{
        "Task": t["task"],
        "Course": t["course"],
        "Type": t["type"],
        "Deadline": t["deadline"],
        "Effort (days)": t["effort_days"],
        "AI said": "" if t.get("ai_effort_days") is None else f"{t['ai_effort_days']} d",
        "Why (AI)": t["effort_reason"],
        "Location": t["location"],
        "Evidence": t.get("evidence", ""),
    } for t in tasks]
    df = pd.DataFrame(rows, columns=EDITOR_COLS, index=[t["id"] for t in tasks])
    df["Deadline"] = pd.to_datetime(df["Deadline"])
    return df


def df_to_tasks(df: pd.DataFrame, old_tasks: list) -> list:
    old = {t["id"]: t for t in old_tasks}
    out = []
    for idx, row in df.iterrows():
        tid = None
        try:
            if idx is not None and not pd.isna(idx):
                tid = int(idx)
        except (TypeError, ValueError):
            tid = None
        if tid in old:
            t = dict(old[tid])
        else:  # row added by hand in the table
            t = {"id": new_id(), "ai_effort_days": None, "evidence": ""}
        t["task"] = clean_str(row["Task"]) or "(untitled task)"
        t["course"] = clean_str(row["Course"])
        t["type"] = clean_type(row["Type"])
        dl = row["Deadline"]
        t["deadline"] = None if pd.isna(dl) else pd.Timestamp(dl).to_pydatetime().replace(second=0, microsecond=0)
        t["effort_days"] = clean_effort(row["Effort (days)"])
        t["effort_reason"] = clean_str(row["Why (AI)"]) or ("Added by you" if t.get("ai_effort_days") is None else "")
        t["location"] = clean_str(row["Location"])
        out.append(t)
    return out


# ----------------------------------------------------------------------------
# Calendar export (.ics), plain string, no extra library
# ----------------------------------------------------------------------------
ICS_VTIMEZONE = [
    "BEGIN:VTIMEZONE", "TZID:Europe/Budapest",
    "BEGIN:DAYLIGHT", "TZOFFSETFROM:+0100", "TZOFFSETTO:+0200", "TZNAME:CEST",
    "DTSTART:19700329T020000", "RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=-1SU", "END:DAYLIGHT",
    "BEGIN:STANDARD", "TZOFFSETFROM:+0200", "TZOFFSETTO:+0100", "TZNAME:CET",
    "DTSTART:19701025T030000", "RRULE:FREQ=YEARLY;BYMONTH=10;BYDAY=-1SU", "END:STANDARD",
    "END:VTIMEZONE",
]


def ics_escape(text: str) -> str:
    return (str(text).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n"))


def ics_fold(line: str) -> str:
    """Fold lines longer than 75 octets (RFC 5545), never splitting a UTF-8 character."""
    out, cur, size = [], "", 0
    for ch in line:
        b = len(ch.encode("utf-8"))
        if size + b > 74:
            out.append(cur)
            cur, size = " ", 1
        cur += ch
        size += b
    out.append(cur)
    return "\r\n".join(out)


def build_ics(tasks: list, statuses: dict) -> str:
    stamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Deadline Guardian//GDG Debrecen 2026//EN",
             "CALSCALE:GREGORIAN", "METHOD:PUBLISH", *ICS_VTIMEZONE]
    for t in tasks:
        if t["deadline"] is None:
            continue
        sb = statuses[t["id"]]["start_by"]
        dl = t["deadline"]
        desc = (f"{t['effort_days']} day(s) of work. Deadline: {dl.strftime('%a %d %b %Y, %H:%M')}. "
                f"Location: {t['location'] or '-'}")
        lines += [
            "BEGIN:VEVENT",
            f"UID:{t['id']}-start@deadline-guardian",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{sb.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{(sb + timedelta(days=1)).strftime('%Y%m%d')}",
            f"SUMMARY:{ics_escape('🛡️ Start: ' + t['task'])}",
            f"DESCRIPTION:{ics_escape(desc)}",
            "END:VEVENT",
            "BEGIN:VEVENT",
            f"UID:{t['id']}-deadline@deadline-guardian",
            f"DTSTAMP:{stamp}",
            f"DTSTART;TZID=Europe/Budapest:{dl.strftime('%Y%m%dT%H%M00')}",
            f"DTEND;TZID=Europe/Budapest:{(dl + timedelta(minutes=30)).strftime('%Y%m%dT%H%M00')}",
            f"SUMMARY:{ics_escape('⏰ Deadline: ' + t['task'])}",
            f"DESCRIPTION:{ics_escape(desc)}",
        ]
        if t["location"]:
            lines.append(f"LOCATION:{ics_escape(t['location'])}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(ics_fold(l) for l in lines) + "\r\n"


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
def load_sample(path: Path):
    st.session_state.input_text = path.read_text(encoding="utf-8")


def on_missed_today():
    tasks = st.session_state.tasks
    cur = planning_today()
    before = snapshot(tasks, cur)
    after = snapshot(tasks, cur + timedelta(days=1))
    st.session_state.missed_days += 1
    st.session_state.missed_explanation = explain_missed_day(tasks, before, after)


def on_reset_missed():
    st.session_state.missed_days = 0
    st.session_state.missed_explanation = None


def on_clear_all():
    st.session_state.tasks = []
    st.session_state.editor_version += 1
    st.session_state.missed_explanation = None


# ----------------------------------------------------------------------------
# Styling: one status palette everywhere + light CSS (no fonts, images or JS)
# ----------------------------------------------------------------------------
STATUS_COLORS = {"green": "#16A34A", "yellow": "#D97706", "red": "#DC2626", "grey": "#94A3B8"}
STATUS_TINTS = {"green": "#F0FDF4", "yellow": "#FFFBEB", "red": "#FEF2F2", "grey": "#F8FAFC"}

st.markdown("""<style>
.stApp {background:#FFFFFF;}
.block-container, [data-testid="stMainBlockContainer"] {max-width:1200px; margin:0 auto; padding-top:1.5rem;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#FFFFFF; border:1px solid #E2E8F0 !important;
  border-radius:12px !important; box-shadow:0 1px 3px rgba(15,23,42,0.06);}
h1, h2, h3, h4 {color:#0F172A !important; font-weight:600 !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#64748B !important;}
button[kind="primary"], [data-testid="stBaseButton-primary"] {background:#4F46E5 !important;
  color:#FFFFFF !important; border:none !important; border-radius:8px !important;}
[data-testid="stAppDeployButton"], [data-testid="stDeployButton"], footer {display:none !important;}
.dg-title {font-size:1.9rem; font-weight:700; color:#0F172A; line-height:1.2;}
.dg-sub {color:#64748B; margin-top:.3rem;}
.dg-badge {text-align:right; color:#64748B; font-size:.8rem; padding-top:.9rem;}
.dg-focus {border-left:6px solid; border-radius:10px; padding:.9rem 1.1rem; font-size:1.1rem;
  color:#0F172A; margin-bottom:.75rem;}
.dg-metric {color:#64748B; font-size:.85rem; display:flex; align-items:center; gap:.45rem;}
.dg-dot {width:.65rem; height:.65rem; border-radius:50%; display:inline-block;}
.dg-metric-value {font-size:1.8rem; font-weight:600; color:#0F172A; line-height:1.3;}
.dg-spacer {height:.6rem;}
</style>""", unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


def spacer():
    st.markdown('<div class="dg-spacer"></div>', unsafe_allow_html=True)


def focus_card(code: str, html_text: str):
    st.markdown(f'<div class="dg-focus" style="border-left-color:{STATUS_COLORS[code]};'
                f'background:{STATUS_TINTS[code]}">{html_text}</div>', unsafe_allow_html=True)


def metric_card(col, label: str, value, color: str):
    with col.container(border=True):
        st.markdown(f'<div class="dg-metric"><span class="dg-dot" style="background:{color}"></span>'
                    f'{esc(label)}</div><div class="dg-metric-value">{esc(value)}</div>',
                    unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
api_key = get_api_key()

with st.sidebar:
    st.header("🛡️ Deadline Guardian")
    st.markdown(f"**Planning from:** {fmt_day(planning_today())} {planning_today().year}")
    if st.session_state.missed_days:
        st.caption(f"Includes {st.session_state.missed_days} missed day(s).")
        st.button("↺ Undo missed days", on_click=on_reset_missed)

    with st.expander("🧪 Demo: change today's date"):
        st.date_input("Pretend today is", value=real_today(), key="pretend_today", format="YYYY-MM-DD")
        st.caption(f"Real date (Budapest): {real_today().isoformat()}. Only for demos; it changes "
                   "the planning date, not the AI output.")

    st.divider()
    travel_min = st.number_input("🚋 Travel time to campus (minutes)", min_value=0, max_value=240,
                                 value=30, step=5, key="travel_min")
    ready_min = st.number_input("⏰ Time to get ready (minutes)", min_value=0, max_value=240,
                                value=45, step=5, key="ready_min")
    home = st.text_input("🏠 Your home address or stop", value="Debrecen, Nagyállomás", key="home")

    st.divider()
    st.markdown("**The rule**")
    st.code("start_by = deadline − effort_days − 1 buffer day", language=None)
    st.markdown("🟢 before start-by · 🟡 start-by is today · 🔴 past start-by")
    st.divider()
    st.button("🗑️ Clear all tasks", on_click=on_clear_all, disabled=not st.session_state.tasks)
    st.caption(f"AI: {MODEL_LABEL} extracts tasks and estimates effort. "
               "Statuses, overload and the plan come from fixed rules, not AI. "
               "Nothing is stored; refresh = fresh start.")

# ----------------------------------------------------------------------------
# Header (compact hero)
# ----------------------------------------------------------------------------
h_left, h_right = st.columns([3, 1])
h_left.markdown(
    '<div class="dg-title">🛡️ Deadline Guardian</div>'
    '<div class="dg-sub">Paste your syllabus or a Neptun message. I estimate how long each task '
    '<b>really</b> takes and warn you <b>early enough to finish it</b>, not the day before.</div>',
    unsafe_allow_html=True)
h_right.markdown('<div class="dg-badge">GDG on Campus Debrecen · Agentic AI Hackathon 2026</div>',
                 unsafe_allow_html=True)
if base_today() != real_today():
    st.warning(f"🧪 Demo date active: planning as if today were {base_today().isoformat()} "
               f"(real date {real_today().isoformat()}).")
spacer()

# ----------------------------------------------------------------------------
# 1. Input + extraction
# ----------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("① Paste your deadlines")
    cols = st.columns(len(SAMPLES) + 1)
    for col, (label, path) in zip(cols, SAMPLES.items()):
        col.button(f"Load example: {label}", on_click=load_sample, args=(path,))

    st.text_area("Syllabus, Neptun message or teacher email (English or Hungarian)",
                 key="input_text", height=220,
                 placeholder="e.g. 'Második ZH: 2026. november 30. 8:00, IK-205' or 'Project report due 11 Dec 23:59'")

    c1, c2 = st.columns([1, 3])
    replace = c2.checkbox("Replace my current list (otherwise new tasks are added)", value=False)

    if not api_key:
        st.error("No `GEMINI_API_KEY` found, so AI extraction is **off** (nothing is faked). "
                 "Set the environment variable or add it to Streamlit secrets. "
                 "You can still add tasks by hand in the table below.")

    if c1.button("✨ Extract with Gemini", type="primary",
                 disabled=(not api_key) or not st.session_state.input_text.strip()):
        try:
            with st.spinner("Gemini is reading your text and estimating the work… (can take up to a minute)"):
                raw, model_used = call_gemini(st.session_state.input_text.strip(), base_today().isoformat(), api_key)
        except Exception as e:  # show the real error, never a fake result
            st.error(f"Gemini call failed: {str(e)[:1500]}")
        else:
            new_tasks = [task_from_ai(r) for r in raw if isinstance(r, dict)]
            existing = [] if replace else list(st.session_state.tasks)
            seen = {(t["task"].lower(), t["deadline"]) for t in existing}
            added, dupes = [], 0
            for t in new_tasks:
                k = (t["task"].lower(), t["deadline"])
                if k in seen:
                    dupes += 1
                    continue
                seen.add(k)
                added.append(t)
            st.session_state.tasks = existing + added
            st.session_state.editor_version += 1
            st.session_state.missed_explanation = None
            msg = f"Gemini ({model_used}) found {len(new_tasks)} task(s); added {len(added)}."
            if dupes:
                msg += f" Skipped {dupes} already in your list."
            no_date = sum(1 for t in added if t["deadline"] is None)
            if no_date:
                msg += f" {no_date} have no date: add it in the table."
            st.session_state.flash = ("success" if new_tasks else "warning",
                                      msg if new_tasks else "Gemini found no graded tasks with dates in this text.")
            st.rerun()

    if st.session_state.flash:
        kind, msg = st.session_state.flash
        getattr(st, kind)(msg)
        st.session_state.flash = None
spacer()

# ----------------------------------------------------------------------------
# 2. Review / edit estimates
# ----------------------------------------------------------------------------
tasks = st.session_state.tasks
with st.container(border=True):
    st.subheader("② Check the AI's estimates")
    st.caption("The AI can be wrong. Fix any cell: change **Effort (days)** if you know you're faster or slower. "
               "Add or delete rows if needed. Everything below updates instantly.")

    df = tasks_to_df(tasks)
    edited = st.data_editor(
        df,
        key=f"editor_{st.session_state.editor_version}",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Task": st.column_config.TextColumn(required=True),
            "Course": st.column_config.TextColumn(),
            "Type": st.column_config.SelectboxColumn(options=TYPES, required=True, default="assignment"),
            "Deadline": st.column_config.DatetimeColumn(format="ddd D MMM YYYY, HH:mm", step=60),
            "Effort (days)": st.column_config.NumberColumn(min_value=1, max_value=60, step=1, format="%d",
                                                           default=1, help="Minimum days of work. Edit me!"),
            "AI said": st.column_config.TextColumn(disabled=True, help="Gemini's original estimate"),
            "Why (AI)": st.column_config.TextColumn(width="large"),
            "Location": st.column_config.TextColumn(),
            "Evidence": st.column_config.TextColumn(disabled=True, width="medium",
                                                    help="Quote from your text where Gemini found it"),
        },
    )
    new_tasks = df_to_tasks(edited, tasks)
    if signature(new_tasks) != signature(tasks):
        st.session_state.tasks = new_tasks
        st.session_state.editor_version += 1
        st.rerun()

    if not tasks:
        st.info("No tasks yet. Load an example or paste your own text, then click **Extract with Gemini**.")
        st.stop()
spacer()

# ----------------------------------------------------------------------------
# 3. Early warnings (the key feature) + 5. I missed today
# ----------------------------------------------------------------------------
today = planning_today()
statuses = {t["id"]: task_status(t, today) for t in tasks}
by_id = {t["id"]: t for t in tasks}

ORDER = {"red": 0, "yellow": 1, "green": 2, "nodate": 3, "passed": 4}


def effort_label(t):
    ai = t.get("ai_effort_days")
    if ai is not None and ai != t["effort_days"]:
        return f"{t['effort_days']} d (AI: {ai})"
    return f"{t['effort_days']} d"


def leave_wake(t):
    """Leave-by / wake-up time for exams and presentations, from the user's own travel time."""
    if t["type"] not in ("exam", "presentation") or t["deadline"] is None:
        return "—"
    leave = t["deadline"] - timedelta(minutes=int(travel_min) + 10)
    wake = leave - timedelta(minutes=int(ready_min))
    return f"Leave {leave:%H:%M} · Wake {wake:%H:%M}"


with st.container(border=True):
    st.subheader("③ Your early warnings")

    # Today's focus: one big message for the most urgent task
    _red = sorted([t for t in tasks if statuses[t["id"]]["code"] == "red"], key=lambda t: t["deadline"])
    _yellow = sorted([t for t in tasks if statuses[t["id"]]["code"] == "yellow"], key=lambda t: t["deadline"])
    _green = sorted([t for t in tasks if statuses[t["id"]]["code"] == "green"],
                    key=lambda t: statuses[t["id"]]["start_by"])
    if _red:
        _t = _red[0]
        _n = statuses[_t["id"]]["late_days"]
        focus_card("red", f"🔴 <b>{esc(_t['task'])}</b>: you are {_n} day{'s' if _n != 1 else ''} "
                          f"late to start. Start it now.")
    elif _yellow:
        _t = _yellow[0]
        focus_card("yellow", f"🟡 Today: start <b>{esc(_t['task'])}</b>. {esc(statuses[_t['id']]['message'])}")
    elif _green:
        _t = _green[0]
        focus_card("green", f"🟢 All on track. Next start-by: <b>{esc(_t['task'])}</b> on "
                            f"{fmt_day(statuses[_t['id']]['start_by'])}.")
    else:
        focus_card("grey", "No upcoming deadlines with dates.")

    codes = [s["code"] for s in statuses.values()]
    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "At risk", codes.count("red"), STATUS_COLORS["red"])
    metric_card(m2, "Start today", codes.count("yellow"), STATUS_COLORS["yellow"])
    metric_card(m3, "On track", codes.count("green"), STATUS_COLORS["green"])
    metric_card(m4, "Planning date", fmt_day(today), "#4F46E5")

    b1, b2 = st.columns([1, 3])
    b1.button("😴 I missed today", on_click=on_missed_today,
              help="Didn't work today? Recalculate everything from tomorrow and see what changed.")
    if st.session_state.missed_days:
        b2.button("↺ Undo missed days", on_click=on_reset_missed, key="undo_main")
    if st.session_state.missed_explanation:
        st.info("**What changed after 'I missed today':**\n\n" +
                "\n".join(f"- {line}" for line in st.session_state.missed_explanation))

    rows = []
    for t in sorted(tasks, key=lambda t: (ORDER[statuses[t["id"]]["code"]],
                                          statuses[t["id"]]["start_by"] or date.max)):
        s = statuses[t["id"]]
        rows.append({
            "Status": s["status"],
            "Task": f"{TYPE_ICON.get(t['type'], '')} {t['task']}",
            "Course": t["course"],
            "Deadline": t["deadline"].strftime("%a %d %b %Y, %H:%M") if t["deadline"] else "—",
            "Effort": effort_label(t),
            "Start by": fmt_day(s["start_by"]) if s["start_by"] else "—",
            "What to do": s["message"],
            "Location": t["location"],
            "Leave by / Wake up": leave_wake(t),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"What to do": st.column_config.TextColumn(width="large")})
    st.caption("Unlike a calendar, which only alerts on the day (or when you set an alarm), "
               "this warning is automatic and based on how much work the task needs.")
    st.caption("Leave by / Wake up: based on your own travel time. Live DKV tram/bus timetables are a next step.")

    # Trip lines: next 3 upcoming exams/presentations with a location, soonest first
    trips = sorted([t for t in tasks if t["type"] in ("exam", "presentation") and t["deadline"]
                    and t["location"] and t["deadline"].date() >= today], key=lambda t: t["deadline"])[:3]
    for t in trips:
        dl = t["deadline"]
        arrive = dl - timedelta(minutes=10)
        leave = dl - timedelta(minutes=int(travel_min) + 10)
        wake = leave - timedelta(minutes=int(ready_min))
        maps_url = ("https://www.google.com/maps/dir/?api=1"
                    f"&origin={urllib.parse.quote_plus(home or 'Debrecen')}"
                    f"&destination={urllib.parse.quote_plus(t['location'] + ', University of Debrecen, Debrecen')}"
                    "&travelmode=transit")
        tc1, tc2 = st.columns([3, 1])
        tc1.markdown(f"{TYPE_ICON.get(t['type'], '📝')} **{t['task']}** · {fmt_day(dl.date())} {dl:%H:%M} at "
                     f"{t['location']} → arrive by {arrive:%H:%M} · Leave {leave:%H:%M} · Wake {wake:%H:%M}")
        tc2.link_button("🚋 Live trams & buses (Google Maps)", maps_url)
    if trips:
        st.caption("Leave/wake times use your own travel time. The button opens Google Maps, which shows DKV's "
                   "real-time departures. Direct DKV data integration (GTFS-RT) is a next step.")

    if any(t["deadline"] for t in tasks):
        st.download_button("📅 Add warnings to my calendar (.ics)", data=build_ics(tasks, statuses),
                           file_name="deadline-guardian.ics", mime="text/calendar")
        st.caption("Imports into Google Calendar / Outlook. The reminder lands on the day you need to START, "
                   "not the deadline.")

    # Timeline chart: bar from start-by date to deadline, dashed line = planning date
    chart_rows, labels_seen = [], set()
    for t in sorted(tasks, key=lambda t: statuses[t["id"]]["start_by"] or date.max):
        s = statuses[t["id"]]
        if s["code"] not in ("red", "yellow", "green"):
            continue
        label = f"{t['task']} ({t['course']})" if t["course"] else t["task"]
        if label in labels_seen:
            label = f"{label} #{t['id']}"
        labels_seen.add(label)
        chart_rows.append({
            "Task": label,
            "Start by": f"{s['start_by'].isoformat()}T00:00:00",
            "Deadline": t["deadline"].strftime("%Y-%m-%dT%H:%M:00"),
            "Status": s["status"],
            "Effort": f"{t['effort_days']} day(s)",
            "Advice": s["message"],
        })

    if chart_rows:
        cdf = pd.DataFrame(chart_rows)
        order = list(cdf["Task"])
        bars = alt.Chart(cdf).mark_bar(cornerRadius=4, height=16).encode(
            x=alt.X("Start by:T", title=None, axis=alt.Axis(format="%a %d %b", labelAngle=0)),
            x2="Deadline:T",
            y=alt.Y("Task:N", sort=order, title=None, axis=alt.Axis(labelLimit=380, labelOverlap=False)),
            color=alt.Color("Status:N",
                            scale=alt.Scale(domain=[STATUS_ON_TRACK, STATUS_START_TODAY, STATUS_AT_RISK],
                                            range=[STATUS_COLORS["green"], STATUS_COLORS["yellow"],
                                                   STATUS_COLORS["red"]]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Task", "Status", "Effort", "Advice",
                     alt.Tooltip("Start by:T", format="%a %d %b"),
                     alt.Tooltip("Deadline:T", format="%a %d %b %H:%M")],
        )
        today_rule = alt.Chart(pd.DataFrame({"x": [f"{today.isoformat()}T00:00:00"]})).mark_rule(
            strokeDash=[5, 4], color="#64748B", size=2).encode(x="x:T")
        st.altair_chart((bars + today_rule).properties(height=max(160, 45 * len(cdf))))
        st.caption("Each bar runs from the start-by date to the deadline. Dashed line = planning date. "
                   "A bar starting left of the line means you're late to start.")
spacer()

# ----------------------------------------------------------------------------
# 4. Overload detection
# ----------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("④ Overload check")
    ranges = detect_overload(tasks, today)
    if ranges:
        for r in ranges:
            span = fmt_day(r["start"]) if r["start"] == r["end"] else f"{fmt_day(r['start'])} – {fmt_day(r['end'])}"
            names = ", ".join(f"**{by_id[i]['task']}**" for i in r["task_ids"])
            st.warning(f"⚠️ {span}: {len(r['task_ids'])} tasks need work at the same time: {names}")
    else:
        st.success("No overlapping work days. Each task has its own time. 👍")

    tips = advice(tasks, today)
    if tips:
        st.markdown("**What to do first**")
        st.markdown("\n".join(f"- {tip}" for tip in tips))

    plan = build_plan(tasks, today)
    if plan["days"]:
        with st.expander("📅 Day-by-day plan (one task per day where possible)", expanded=bool(ranges)):
            deadlines_by_day = {}
            for t in tasks:
                if t["deadline"] and t["deadline"].date() >= today:
                    deadlines_by_day.setdefault(t["deadline"].date(), []).append(t["task"])
            all_days = sorted(set(plan["days"]) | set(deadlines_by_day))
            plan_rows = [{
                "Day": fmt_day(d) + (" (today)" if d == today else ""),
                "Work on": " + ".join(by_id[i]["task"] for i in plan["days"].get(d, [])),
                "Load": "⚠️ shared day" if len(plan["days"].get(d, [])) > 1 else "",
                "Deadlines": ", ".join(deadlines_by_day.get(d, [])),
            } for d in all_days]
            st.dataframe(pd.DataFrame(plan_rows), hide_index=True)
            st.caption("Tasks with later deadlines are placed as late as possible, so the earlier ones get "
                       "the earlier days. Free days are your extra slack.")

st.divider()
st.caption(f"Built at GDG on Campus Debrecen Agentic AI Hackathon, 21 Nov 2026. "
           f"Task extraction and effort estimates: {MODEL_LABEL} (check them!). "
           f"Warnings and plan: deterministic rules in planner.py.")
