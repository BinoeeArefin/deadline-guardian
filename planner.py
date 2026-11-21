"""
Deadline Guardian - planning logic (no AI, no Streamlit).

Everything here is plain, deterministic Python so the warnings are
explainable: the AI only extracts tasks and estimates effort; the rules
below decide the status, the overload flags and the day-by-day plan.

A task is a dict with at least:
    id, task, course, type, deadline (datetime | None), effort_days (int)
"""

from collections import defaultdict
from datetime import date, datetime, timedelta

BUFFER_DAYS = 1  # one safety day between finishing the work and the deadline

# Used only to break ties when two tasks have the same start-by date.
TYPE_PRIORITY = {"exam": 0, "project": 1, "presentation": 2, "assignment": 3}

STATUS_ON_TRACK = "🟢 On track"
STATUS_START_TODAY = "🟡 Start today"
STATUS_AT_RISK = "🔴 At risk"
STATUS_PASSED = "⚫ Deadline passed"
STATUS_NO_DATE = "⚪ No date"


def fmt_day(d: date) -> str:
    """'Mon 23 Nov' (cross-platform, no %-d)."""
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b')}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def effort_of(task: dict) -> int:
    try:
        return max(1, int(task.get("effort_days") or 1))
    except (TypeError, ValueError):
        return 1


def start_by_date(deadline: datetime, effort_days: int) -> date:
    """THE early-warning rule: start_by = deadline - effort_days - 1 buffer day."""
    return deadline.date() - timedelta(days=effort_days + BUFFER_DAYS)


def task_status(task: dict, today: date) -> dict:
    """Status of one task relative to `today` (the planning date)."""
    deadline = task.get("deadline")
    if deadline is None:
        return {
            "status": STATUS_NO_DATE,
            "code": "nodate",
            "start_by": None,
            "days_left": None,
            "late_days": 0,
            "message": "No deadline found. Add one in the table to get a warning.",
        }

    effort = effort_of(task)
    sb = start_by_date(deadline, effort)
    days_left = (deadline.date() - today).days

    if deadline.date() < today:
        return {
            "status": STATUS_PASSED,
            "code": "passed",
            "start_by": sb,
            "days_left": days_left,
            "late_days": 0,
            "message": f"Deadline was {fmt_day(deadline.date())}.",
        }

    if today < sb:
        n = (sb - today).days
        return {
            "status": STATUS_ON_TRACK,
            "code": "green",
            "start_by": sb,
            "days_left": days_left,
            "late_days": 0,
            "message": f"Start by {fmt_day(sb)} (in {plural(n, 'day')}).",
        }

    if today == sb:
        return {
            "status": STATUS_START_TODAY,
            "code": "yellow",
            "start_by": sb,
            "days_left": days_left,
            "late_days": 0,
            "message": f"Start today: {plural(effort, 'day')} of work + 1 buffer day before the deadline.",
        }

    late = (today - sb).days
    if days_left == 0:
        extra = "Due today."
    else:
        extra = f"Only {plural(days_left, 'day')} left for {plural(effort, 'day')} of work."
    return {
        "status": STATUS_AT_RISK,
        "code": "red",
        "start_by": sb,
        "days_left": days_left,
        "late_days": late,
        "message": f"You are {plural(late, 'day')} late to start. {extra}",
    }


def active_tasks(tasks: list, today: date) -> list:
    """Tasks that still need work: they have a date that is today or later."""
    return [t for t in tasks if t.get("deadline") is not None and t["deadline"].date() >= today]


def work_window(task: dict, today: date) -> list:
    """Days this task needs work if it were your only task.

    Normal case: start_by .. deadline-2 (deadline-1 is the buffer day).
    If you're already late: today .. deadline-2 (squeezed), never empty.
    """
    deadline_day = task["deadline"].date()
    sb = start_by_date(task["deadline"], effort_of(task))
    start = max(sb, today)
    end = deadline_day - timedelta(days=BUFFER_DAYS + 1)
    if end < start:
        end = start  # deadline today/tomorrow: at least today counts
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def detect_overload(tasks: list, today: date) -> list:
    """Find days where 2+ tasks need work at the same time.

    Returns a list of ranges: {"start", "end", "task_ids"} with consecutive
    days that share the same set of tasks merged together.
    """
    load = defaultdict(list)
    for t in active_tasks(tasks, today):
        for d in work_window(t, today):
            load[d].append(t["id"])

    busy_days = sorted(d for d, ids in load.items() if len(ids) >= 2)
    ranges = []
    for d in busy_days:
        ids = sorted(load[d])
        if ranges and ranges[-1]["task_ids"] == ids and ranges[-1]["end"] + timedelta(days=1) == d:
            ranges[-1]["end"] = d
        else:
            ranges.append({"start": d, "end": d, "task_ids": ids})
    return ranges


def priority_key(task: dict, today: date):
    sb = start_by_date(task["deadline"], effort_of(task))
    return (sb, task["deadline"], TYPE_PRIORITY.get(task.get("type", ""), 9), -effort_of(task))


def build_plan(tasks: list, today: date) -> dict:
    """Load-aware plan: ideally ONE task per day.

    Tasks with later deadlines are placed first, each as late as possible
    before its own buffer day, so earlier deadlines get the earlier days.
    If a task doesn't fit:
      1. it uses its buffer day,
      2. then it shares a day with another task (= real overload),
      3. anything still missing (not enough calendar days left) is a shortfall.

    Returns {"days": {date: [task_id, ...]}, "plan_start": {id: date},
             "shortfall": {id: int}, "used_buffer": set(ids),
             "shared": {id: [dates]}}
    """
    days = defaultdict(list)
    plan_start, shortfall, used_buffer, shared = {}, {}, set(), {}

    todo = sorted(active_tasks(tasks, today), key=lambda t: t["deadline"], reverse=True)
    for t in todo:
        tid = t["id"]
        need = effort_of(t)
        deadline_day = t["deadline"].date()
        last_work = deadline_day - timedelta(days=BUFFER_DAYS + 1)
        buffer_day = deadline_day - timedelta(days=BUFFER_DAYS)
        mine = set()

        # 1) free normal days, latest first
        d = last_work
        while need > 0 and d >= today:
            if not days[d]:
                days[d].append(tid); mine.add(d); need -= 1
            d -= timedelta(days=1)

        # 2) free buffer day (or today, if the deadline is today)
        extra_day = today if deadline_day == today else buffer_day
        for d in (extra_day,):
            if need > 0 and today <= d <= deadline_day and d not in mine and not days[d]:
                days[d].append(tid); mine.add(d); used_buffer.add(tid); need -= 1

        # 3) share the least-busy remaining days (latest first)
        if need > 0:
            end = buffer_day if deadline_day > today else today
            candidates = []
            d = end
            while d >= today:
                if d not in mine:
                    candidates.append(d)
                d -= timedelta(days=1)
            candidates.sort(key=lambda x: len(days[x]))  # stable: keeps latest-first among equals
            for d in candidates[:need]:
                days[d].append(tid); mine.add(d); need -= 1
                shared.setdefault(tid, []).append(d)
                if d == buffer_day:
                    used_buffer.add(tid)

        if mine:
            plan_start[tid] = min(mine)
        if need > 0:
            shortfall[tid] = need

    clean_days = {d: ids for d, ids in sorted(days.items()) if ids}
    return {"days": clean_days, "plan_start": plan_start, "shortfall": shortfall,
            "used_buffer": used_buffer, "shared": shared}


def advice(tasks: list, today: date) -> list:
    """Plain-language suggestions: what to start first and what to move earlier."""
    by_id = {t["id"]: t for t in tasks}
    act = active_tasks(tasks, today)
    if not act:
        return []
    plan = build_plan(tasks, today)
    out = []

    def name(tid):
        return f"**{by_id[tid]['task']}**"

    today_ids = plan["days"].get(today, [])
    if today_ids:
        # earliest deadline first if two tasks share today
        today_ids = sorted(today_ids, key=lambda i: by_id[i]["deadline"])
        out.append("**Today, work on:** " + " → then ".join(name(i) for i in today_ids) + ".")
    else:
        nxt = next(iter(plan["days"].items()), None)
        if nxt:
            out.append(f"Nothing *has* to be done today. Next planned work: {name(nxt[1][0])} on "
                       f"{fmt_day(nxt[0])}. Starting it today gives you extra slack.")

    first = sorted(act, key=lambda t: priority_key(t, today))[0]
    if first["id"] not in today_ids:
        out.append(f"Earliest start-by date: {name(first['id'])} "
                   f"({fmt_day(start_by_date(first['deadline'], effort_of(first)))}).")

    for t in sorted(act, key=lambda t: priority_key(t, today)):
        tid = t["id"]
        sb = start_by_date(t["deadline"], effort_of(t))
        ps = plan["plan_start"].get(tid)
        if ps is not None and today <= ps < sb:
            out.append(f"Because of overlaps, start {name(tid)} on {fmt_day(ps)} instead of {fmt_day(sb)}.")
        if tid in plan["shortfall"]:
            n = plan["shortfall"][tid]
            out.append(f"⚠️ {name(tid)} doesn't fit: {plural(n, 'day')} of work missing even if you work every day. "
                       f"Cut scope, team up, or talk to the teacher now, not the day before.")
        elif tid in plan["shared"]:
            ds = ", ".join(fmt_day(d) for d in sorted(plan["shared"][tid]))
            out.append(f"⚠️ {name(tid)} has to share {ds} with another task. Do the one due first, first thing.")
        elif tid in plan["used_buffer"]:
            out.append(f"{name(tid)} only fits if you use its buffer day.")
    return out


def snapshot(tasks: list, today: date) -> dict:
    """Everything the 'I missed today' diff compares."""
    plan = build_plan(tasks, today)
    return {
        "today": today,
        "status": {t["id"]: task_status(t, today) for t in tasks},
        "overload_days": {d for r in detect_overload(tasks, today)
                          for d in _daterange(r["start"], r["end"])},
        "plan": plan,
    }


def _daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def explain_missed_day(tasks: list, before: dict, after: dict) -> list:
    """Human explanation of what changed after the user missed a day."""
    by_id = {t["id"]: t for t in tasks}
    lines = [f"You lost {fmt_day(before['today'])}. Your plan now starts on **{fmt_day(after['today'])}**."]

    lost = [i for i in before["plan"]["days"].get(before["today"], []) if i in by_id]
    if lost:
        names = " and ".join(f"**{by_id[i]['task']}**" for i in lost)
        lines.append(f"The day planned for {names} is gone, so that work moves to later days.")

    changed = 0
    for tid, new in after["status"].items():
        old = before["status"].get(tid)
        if old is None or tid not in by_id:
            continue
        name = by_id[tid]["task"]
        if old["status"] != new["status"]:
            lines.append(f"**{name}**: {old['status']} → {new['status']}. {new['message']}")
            changed += 1
        elif new["code"] == "red" and new["late_days"] != old["late_days"]:
            lines.append(f"**{name}**: still at risk, now {plural(new['late_days'], 'day')} late to start.")
            changed += 1

    new_short = {k: v for k, v in after["plan"]["shortfall"].items()
                 if v > before["plan"]["shortfall"].get(k, 0)}
    for tid, n in new_short.items():
        lines.append(f"⚠️ **{by_id[tid]['task']}** no longer fits: {plural(n, 'day')} short even working every day.")
        changed += 1

    added_busy = sorted(after["overload_days"] - before["overload_days"])
    if added_busy:
        lines.append("New overloaded days: " + ", ".join(fmt_day(d) for d in added_busy) + ".")
        changed += 1

    if changed == 0:
        lines.append("No status changed: you had enough slack. 👍 Don't make it a habit.")
    return lines
