# 🛡️ Deadline Guardian
🔗 **Live demo:** https://deadline-guardian-debrecen.streamlit.app (no install needed, just click)

**Deadline Guardian reads your syllabus and university emails, estimates how long each task really takes, and warns you early enough to finish it. Built for University of Debrecen students.**

GDG on Campus Debrecen, Agentic AI Hackathon, 21 November 2026. Track: Smart Campus (also fits Agents).

## The problem

Calendars warn you on the day of the deadline, or when you set an alarm yourself. An exam that needs 3 days of study is already lost if the reminder arrives the evening before. Students also get deadlines in many formats and languages (an English syllabus, a Hungarian Neptun message, a teacher's email), so nothing collects them in one place.

## What it does

1. **Extract.** Paste any text (English or Hungarian). Gemini returns structured JSON for each graded item: task, course, type (exam / assignment / project / presentation), deadline (ISO date and time), location, plus a short quote from your text as evidence.
2. **Estimate effort.** Gemini estimates the minimum days of work each task needs, with a one-line reason (weight %, page count, topics covered…). **You can edit every estimate** in the table. The app shows the AI's original number next to yours.
3. **Early warning (the key feature).** A fixed, explainable rule:

   ```
   start_by = deadline − effort_days − 1 buffer day
   ```
   - 🟢 **On track**: today is before start_by
   - 🟡 **Start today**: today is start_by
   - 🔴 **At risk**: today is after start_by ("you are X days late to start")

   The warning appears automatically and depends on the workload, not on an alarm you remembered to set.
4. **Overload detection.** Flags days where two or more tasks need work at the same time. It then builds a day-by-day plan (one task per day where possible, tasks with later deadlines placed as late as possible) and tells you what to start first and which tasks to move earlier.
5. **"I missed today" button.** Marks today as lost, recalculates every status and the plan from tomorrow, and explains what changed (e.g. "Lab assignment 2: 🟡 Start today → 🔴 At risk").
6. Calendar export (.ics): start-by reminders + deadlines for Google Calendar / Outlook.

The AI handles the language parts (reading messy text, estimating effort). The warnings, overload flags and plan are **deterministic rules** in `planner.py`, so every warning can be explained.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY="your-key"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
$env:GEMINI_MODEL="..."   # optional: override the model if Google retires this one
streamlit run app.py
```

Get a key at <https://aistudio.google.com/apikey>. You can also put it in `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`).

Then click **Load example** (English syllabus or Hungarian Neptun message) → **Extract with Gemini**. Load the second example too: the two together show the overload detection, with two exams on the same Monday.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo (`.streamlit/secrets.toml` is git-ignored).
2. On <https://share.streamlit.io>: **New app**, pick the repo, main file `app.py`.
3. **Advanced settings → Secrets**: `GEMINI_API_KEY = "your-key"`.

## Files

| File | What it is |
|---|---|
| `app.py` | Streamlit UI and the Gemini call: Google Gemini API (gemini-3.7-flash by default, with automatic fallback to other Gemini flash models if one is overloaded; override with GEMINI_MODEL), JSON output with a response schema |
| `planner.py` | Start-by rule, statuses, overload detection, day-by-day plan, "I missed today" diff. No AI. |
| `samples/syllabus_en.txt` | Sample input: English course syllabus (made up for the demo) |
| `samples/neptun_hu.txt` | Sample input: Hungarian Neptun-style teacher message (made up for the demo) |

## Honest limitations

- **Sample inputs are made up; outputs are not.** Every task, date and estimate on screen comes from a live Gemini call on the text you paste. Nothing is hardcoded. If the API key is missing or the call fails, the app shows the real error and does not show fake results. You can still add tasks by hand.
- Effort estimates are AI guesses for a "typical student". Check them and edit them.
- The day-by-day plan is a simplification: one task per full day, deadlines in Budapest time.
- No Neptun/Moodle/email integration yet. You copy and paste the text.
- No login and no database. Data lives only in your browser session; to keep your results, use the **📅 .ics calendar export** (start-by reminders + deadlines).
- Identical text on the same date is cached for 1 hour to save API quota (it's still a real Gemini result).
- The sidebar "Demo: change today's date" option is for presentations. It is clearly labelled in the UI when active.

## Tools, APIs and AI assistants used

- **Google Gemini API (gemini-3.7-flash by default, with automatic fallback to other Gemini flash models if one is overloaded; override with GEMINI_MODEL)**, through the `google-genai` Python library: task extraction and effort estimates
- **Streamlit**: UI, and **Streamlit Community Cloud** for hosting
- **pandas** and **Altair** (bundled with Streamlit): tables and the timeline chart
- **pydantic**: the JSON response schema sent to Gemini
- **Claude** (Anthropic): AI coding assistant used to help write the code and README
- **ChatGPT** (OpenAI): AI assistant used during the hackathon

All code was written on 21 November 2026, during the hackathon.

## Next steps

- Neptun / Moodle / Outlook import instead of copy and paste
- Learn your personal speed from past tasks ("you usually need 1.5× the estimate")
- Push / email reminders on the start-by date
- Leave-by time for exams from the DKV timetable
