"""
advisor_engine.py
-----------------
The "brain" of the AI-Powered Student Academic Advisory Agent.

It is deliberately separated from the user interface so it can be tested and
explained on its own. It combines THREE intelligent techniques:

1. MACHINE LEARNING  - a Ridge regression model predicts a student's final
   score in each subject from continuous-assessment (CA) marks, the CA trend,
   attendance and weekly study hours.  (Trained on synthetic data - see
   `_train_model`; swap in real historical data when available.)
2. RULE-BASED REASONING - an expert-system rule base diagnoses *why* a subject
   is weak (declining trend, inconsistent marks, exam-technique gap, low
   attendance ...) and selects targeted study activities.
3. CONVERSATIONAL AGENT - an intent-matching chatbot that answers questions,
   runs "what-if" simulations and calculates the exam mark needed for a goal.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import html as _html
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
CA_WEIGHT = 0.30      # CATs + assignments contribute 30 %
EXAM_WEIGHT = 0.70    # end-of-semester exam contributes 70 %
CA_COLUMNS = ["CAT1", "CAT2", "Assignment"]
MARK_COLUMNS = CA_COLUMNS + ["Exam"]

GRADE_BANDS = [(70, "A", "Excellent"), (60, "B", "Good"),
               (50, "C", "Average"), (40, "D", "Pass"), (0, "E", "Fail")]

TARGETS = {"a": 70, "first": 70, "b": 60, "second upper": 60, "c": 50,
           "second lower": 50, "d": 40, "pass": 40}


def grade_of(mark: float) -> str:
    for threshold, letter, _ in GRADE_BANDS:
        if mark >= threshold:
            return letter
    return "E"


def grade_label(mark: float) -> str:
    for threshold, _, label in GRADE_BANDS:
        if mark >= threshold:
            return label
    return "Fail"


def classification(avg: float) -> str:
    if avg >= 70:
        return "First Class Honours"
    if avg >= 60:
        return "Second Class Honours (Upper Division)"
    if avg >= 50:
        return "Second Class Honours (Lower Division)"
    if avg >= 40:
        return "Pass"
    return "Below pass mark - at risk"


def _clean(value):
    """Return a float in [0, 100] or None for blank / invalid cells."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    return float(min(100.0, max(0.0, v)))


# --------------------------------------------------------------------------
# 1. MACHINE LEARNING MODEL
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _train_model():
    """Train a Ridge regression on synthetic-but-realistic student records.

    Features : [CA average, CA trend, attendance %, study hours/week]
    Target   : final overall subject mark (0-100)

    The coefficients encode well-documented relationships: CA performance is
    the strongest predictor, an improving trend and good attendance help, and
    study time has diminishing returns beyond ~20 h/week.
    """
    rng = np.random.default_rng(42)
    n = 4000
    ca = np.clip(rng.normal(58, 15, n), 0, 100)
    trend = rng.normal(0, 8, n)
    attendance = rng.uniform(40, 100, n)
    study = rng.uniform(0, 28, n)

    final = (0.70 * ca + 0.35 * trend + 0.10 * (attendance - 75)
             + 0.30 * (np.minimum(study, 20) - 8) + 8
             + rng.normal(0, 6, n))
    final = np.clip(final, 0, 100)

    X = np.column_stack([ca, trend, attendance, study])
    model = Ridge(alpha=1.0).fit(X, final)
    residual_sd = float(np.std(final - model.predict(X)))
    return model, residual_sd


def ml_predict(ca_avg: float, trend: float, attendance: float, study: float):
    """Return (predicted final mark, low, high) with an ~80 % interval."""
    model, sd = _train_model()
    x = np.array([[ca_avg, trend, attendance, study]])
    pred = float(np.clip(model.predict(x)[0], 0, 100))
    margin = 1.28 * sd
    return pred, max(0.0, pred - margin), min(100.0, pred + margin)


# --------------------------------------------------------------------------
# 2. RULE-BASED REASONING + ANALYSIS
# --------------------------------------------------------------------------
def analyse(df: pd.DataFrame, attendance: float = 75, study_hours: float = 10):
    """Analyse every subject and return a list of result dictionaries."""
    results = []
    for _, row in df.iterrows():
        name = str(row.get("Subject", "")).strip()
        if not name or name.lower() == "nan":
            continue
        ca = [_clean(row.get(c)) for c in CA_COLUMNS]
        ca_vals = [v for v in ca if v is not None]
        exam = _clean(row.get("Exam"))
        if not ca_vals and exam is None:
            continue

        ca_avg = float(np.mean(ca_vals)) if ca_vals else float(exam)
        cat1, cat2 = ca[0], ca[1]
        trend = (cat2 - cat1) if (cat1 is not None and cat2 is not None) else 0.0
        spread = (max(ca_vals) - min(ca_vals)) if len(ca_vals) > 1 else 0.0

        if exam is not None:   # final result is known
            overall = CA_WEIGHT * ca_avg + EXAM_WEIGHT * exam if ca_vals else exam
            predicted, low, high = overall, overall, overall
            status = "Final"
        else:                  # ask the ML model to forecast the final mark
            predicted, low, high = ml_predict(ca_avg, trend, attendance, study_hours)
            status = "Predicted"
            overall = None

        res = {
            "subject": name, "ca_avg": ca_avg, "trend": trend, "spread": spread,
            "exam": exam, "overall": overall, "predicted": predicted,
            "low": low, "high": high, "status": status,
            "grade": grade_of(predicted), "label": grade_label(predicted),
        }
        res["reasons"], res["activities"] = _diagnose(res, attendance, study_hours)
        res["risk"] = _risk_level(res)
        res["weak"] = res["risk"] in ("High", "Medium")
        res["priority"] = _priority(res)
        results.append(res)
    return results


def _risk_level(r) -> str:
    p = r["predicted"]
    if p < 40:
        return "High"
    if p < 50 or (r["ca_avg"] < 45):
        return "Medium"
    if p < 60 or r["trend"] < -8:
        return "Watch"
    return "Low"


def _priority(r) -> float:
    """Higher = needs attention sooner (used to rank subjects & share time)."""
    gap = max(0.0, 65 - r["predicted"])
    decline = max(0.0, -r["trend"]) * 0.5
    return round(gap + decline, 1)


def _diagnose(r, attendance, study):
    """Expert-system rules: IF <condition> THEN <reason + study activity>."""
    reasons, acts = [], []

    if r["predicted"] < 40:
        reasons.append("Projected to fall below the 40 % pass mark.")
        acts.append("Book a consultation with your lecturer this week and agree a recovery plan.")
        acts.append("Re-read the whole syllabus and list every topic you cannot explain in your own words.")
    elif r["predicted"] < 50:
        reasons.append("Performance is in the borderline 'Pass' zone (40-49 %).")
        acts.append("Do two timed past-paper questions per day and mark them against the marking scheme.")

    if r["trend"] <= -8:
        reasons.append(f"Marks are DECLINING ({r['trend']:+.0f} points from CAT 1 to CAT 2).")
        acts.append("Review the mistakes in your last CAT script and write a one-page 'error log'.")
    elif r["trend"] >= 8:
        reasons.append(f"Marks are IMPROVING ({r['trend']:+.0f} points) - your current method works.")

    if r["spread"] >= 25:
        reasons.append("Results are inconsistent across assessments (very high spread).")
        acts.append("Use a weekly revision timetable so every topic is revisited, not just exam-week topics.")

    if r["exam"] is not None and r["ca_avg"] - r["exam"] >= 15:
        reasons.append("Exam mark is much lower than coursework - likely exam technique / time pressure.")
        acts.append("Practise under exam conditions: closed-book, timed, and review time allocation.")
    elif r["exam"] is not None and r["exam"] - r["ca_avg"] >= 15:
        reasons.append("Exam outperformed coursework - strong revision, but CA submissions need more care.")

    if r["ca_avg"] < 50 and r["predicted"] >= 40:
        reasons.append("Coursework average is below 50 % - the foundations need strengthening.")
        acts.append("Redo the assignment questions you lost marks on and explain solutions to a classmate.")

    if attendance < 70 and r["predicted"] < 60:
        reasons.append(f"Attendance of {attendance:.0f}% is limiting your result.")
        acts.append("Aim for 85 %+ attendance - missed lectures are strongly linked to lower marks.")

    if study < 8 and r["predicted"] < 60:
        reasons.append(f"Only {study:.0f} study hours per week - below the ~12-15 h typically needed.")
        acts.append("Add two focused 90-minute study blocks per week for this subject.")

    if r["predicted"] >= 70 and not reasons:
        reasons.append("Strong, consistent performance.")
        acts.append("Stretch yourself: attempt past-paper 'hard' questions and help peers (teaching deepens mastery).")
    if not reasons:
        reasons.append("Solid performance with room to move up a grade.")
    if not acts:
        acts.append("Keep a steady weekly revision rhythm and attempt extra practice questions.")
    return reasons, acts


# --------------------------------------------------------------------------
# Learning resources recommender
# --------------------------------------------------------------------------
_OFFLINE = None


def unit_research(subject: str, researcher=None) -> dict:
    """Research a unit by its name (Gemini + Google Search, then Wikipedia/Open Library, then offline matching).
    Works for ANY unit name, listed or not; nothing is picked at random."""
    global _OFFLINE
    if researcher is None:
        if _OFFLINE is None:
            from research import Researcher
            _OFFLINE = Researcher(None, None, get=lambda *a, **k: None, use_web=False)
        researcher = _OFFLINE
    return researcher.research_unit(subject)


def resources_for(subject: str, researcher=None):
    """[(title, url)] for a unit, from research on the unit name."""
    r = unit_research(subject, researcher)
    return [(f"{x['title']} [{x['type']}]" if x.get("type") else x["title"], x["url"]) for x in r["resources"][:6]]


# --------------------------------------------------------------------------
# Study plan + goal calculators
# --------------------------------------------------------------------------
def study_plan(results, weekly_hours: float = 15):
    """Share weekly study hours across subjects in proportion to need."""
    if not results:
        return []
    weights = np.array([r["priority"] + 5 for r in results], dtype=float)
    hours = weights / weights.sum() * weekly_hours
    hours = np.round(hours * 2) / 2     # nearest half-hour
    plan = []
    for r, h in zip(results, hours):
        plan.append({"Subject": r["subject"], "Priority": r["risk"],
                     "Hours / week": float(h),
                     "Focus": r["activities"][0]})
    return sorted(plan, key=lambda p: -p["Hours / week"])


def needed_exam(ca_avg: float, target: float) -> float:
    """Exam mark required to reach `target` overall (0.3 CA + 0.7 Exam)."""
    return (target - CA_WEIGHT * ca_avg) / EXAM_WEIGHT


def overall_summary(results):
    if not results:
        return {}
    preds = [r["predicted"] for r in results]
    avg = float(np.mean(preds))
    return {
        "average": avg, "classification": classification(avg),
        "n_subjects": len(results),
        "n_weak": sum(r["weak"] for r in results),
        "best": max(results, key=lambda r: r["predicted"]),
        "worst": min(results, key=lambda r: r["predicted"]),
    }


# --------------------------------------------------------------------------
# 3. CONVERSATIONAL AGENT
# --------------------------------------------------------------------------
_GENERIC_WORDS = {"systems", "system", "data", "computer", "computing", "introduction", "intro", "analysis", "design", "theory",
                  "applied", "advanced", "management", "engineering", "science", "studies", "fundamentals", "principles",
                  "and", "the", "for", "unit", "course", "subject"}


def _find_subject(question: str, results):
    q = question.lower()
    for r in results:
        if r["subject"].lower() in q:
            return r
    words = set(re.findall(r"[a-z+]{3,}", q))
    best, best_score = None, 0.0
    for r in results:
        name_words = [w for w in re.findall(r"[a-z+]{3,}", r["subject"].lower()) if w not in _GENERIC_WORDS] or \
            re.findall(r"[a-z+]{3,}", r["subject"].lower())
        score = 0.0
        for nw in name_words:
            for w in words:
                score = max(score, difflib.SequenceMatcher(None, nw, w).ratio())
        if score > best_score:
            best, best_score = r, score
    return best if best_score >= 0.8 else None


def _fmt(r):
    return f"**{r['subject']}** - {r['predicted']:.0f}% (Grade {r['grade']})"


def chat_response(question: str, profile: dict, results: list, researcher=None) -> str:
    """Answer a student's question. Understanding is done by advisor_chat (Gemini or a local intent classifier)."""
    import advisor_chat
    return advisor_chat.respond(question, profile, results, researcher)


def _summary_text(profile: dict, results: list, summ: dict) -> str:
    rows = "; ".join(f"{r['subject']} predicted {r['predicted']:.0f}% (risk {r['risk']})" for r in results)
    return (f"Student course: {profile.get('course', 'unknown')}. Attendance {profile.get('attendance', '?')}%, "
            f"study hours {profile.get('study_hours', '?')}/week. Results: {rows}. Average {summ['average']:.0f}%.")


# --------------------------------------------------------------------------
# Personalised report (self-contained HTML - open in a browser and Print > PDF)
# --------------------------------------------------------------------------
def generate_report_html(profile: dict, results: list, researcher=None) -> str:
    e = _html.escape
    summ = overall_summary(results)
    hrs = float(profile.get("study_hours") or 15)
    plan = study_plan(results, hrs)
    today = _dt.date.today().strftime("%d %B %Y")

    rows = ""
    for r in results:
        colour = {"High": "#e5484d", "Medium": "#f5a524", "Watch": "#d9b800", "Low": "#2fb344"}[r["risk"]]
        ex = f"{r['exam']:.0f}" if r["exam"] is not None else "-"
        rows += (f"<tr><td>{e(r['subject'])}</td><td>{r['ca_avg']:.0f}</td><td>{ex}</td>"
                 f"<td><b>{r['predicted']:.0f}</b> <small>({r['status']})</small></td>"
                 f"<td>{r['grade']}</td><td style='color:{colour};font-weight:600'>{r['risk']}</td></tr>")

    detail = ""
    for r in sorted(results, key=lambda r: -r["priority"]):
        res = "".join(f"<li>{e(t)}</li>" for t, _ in resources_for(r["subject"], researcher))
        detail += (f"<div class='card'><h3>{e(r['subject'])} - {r['predicted']:.0f}% "
                   f"(Grade {r['grade']})</h3><p><b>Diagnosis</b></p><ul>"
                   + "".join(f"<li>{e(x)}</li>" for x in r["reasons"])
                   + "</ul><p><b>Recommended actions</b></p><ul>"
                   + "".join(f"<li>{e(x)}</li>" for x in r["activities"])
                   + f"</ul><p><b>Suggested resources</b></p><ul>{res}</ul></div>")

    plan_rows = "".join(f"<tr><td>{e(p['Subject'])}</td><td>{p['Hours / week']:.1f}</td>"
                        f"<td>{e(p['Priority'])}</td></tr>" for p in plan)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Academic Performance Report - {e(profile.get('name',''))}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:30px auto;color:#1c2333;padding:0 18px}}
h1{{color:#2a78d6;margin-bottom:0}} h2{{border-bottom:2px solid #d8d7d2;padding-bottom:4px;color:#0b0b0b}}
table{{border-collapse:collapse;width:100%;margin:10px 0}} th,td{{border:1px solid #dfe3ee;padding:8px;text-align:left}}
th{{background:#f1f1ee}} .meta{{background:#f6f6f4;padding:12px 16px;border-radius:10px}}
.kpi{{display:flex;gap:12px;margin:14px 0}} .kpi div{{flex:1;background:#2a78d6;color:#fff;padding:14px;border-radius:12px;text-align:center}}
.kpi b{{font-size:26px;display:block}} .card{{border:1px solid #dfe3ee;border-radius:10px;padding:4px 16px;margin:12px 0}}
small{{color:#667}} @media print{{body{{margin:0}}}}
</style></head><body>
<h1>Academic Performance Report</h1><small>Generated {today} by the AI Academic Advisory Agent</small>
<div class="meta"><b>Student:</b> {e(profile.get('name',''))} &nbsp;|&nbsp;
<b>Reg. No:</b> {e(profile.get('reg_no',''))} &nbsp;|&nbsp; <b>Course:</b> {e(profile.get('course',''))}<br>
<b>Attendance:</b> {profile.get('attendance',0):.0f}% &nbsp;|&nbsp; <b>Study hours/week:</b> {hrs:.0f}</div>
<div class="kpi"><div><b>{summ['average']:.0f}%</b>Projected average</div>
<div><b>{summ['classification'].split(' (')[0]}</b>Projected standing</div>
<div><b>{summ['n_weak']}/{summ['n_subjects']}</b>Subjects flagged weak</div></div>
<h2>Subject summary</h2>
<table><tr><th>Subject</th><th>CA avg</th><th>Exam</th><th>Final / predicted</th><th>Grade</th><th>Risk</th></tr>{rows}</table>
<h2>Personalised advice</h2>{detail}
<h2>Weekly study plan</h2>
<table><tr><th>Subject</th><th>Hours / week</th><th>Priority</th></tr>{plan_rows}</table>
<p><small>Predictions come from a regression model plus rule-based reasoning; they are guidance, not guarantees.</small></p>
</body></html>"""


# --------------------------------------------------------------------------
# Sample data (for demos)
# --------------------------------------------------------------------------
def sample_marks() -> pd.DataFrame:
    return pd.DataFrame({
        "Subject": ["Database Systems", "Web Development", "Computer Security",
                    "Data Structures & Algorithms", "Computer Networks", "Discrete Mathematics"],
        "CAT1": [72, 65, 58, 45, 52, 38],
        "CAT2": [78, 70, 52, 38, 55, 30],
        "Assignment": [80, 75, 60, 50, 58, 42],
        "Exam": [None, None, None, None, None, None],
    })


# --------------------------------------------------------------------------
# AUTONOMOUS BEHAVIOUR  (perceive -> analyse -> decide -> recommend -> act)
# --------------------------------------------------------------------------
def draft_consult_email(profile: dict, r: dict) -> tuple[str, str]:
    """Write an email asking the lecturer for help - the agent does this without being asked."""
    who = profile.get("name") or "a student"
    reg = profile.get("reg_no") or "(registration number)"
    reasons = " ".join(r["reasons"][:2])
    subject = f"Request for academic support - {r['subject']} ({reg})"
    body = (f"Dear Lecturer,\n\nI am {who}, registration number {reg}, taking {r['subject']}. "
            f"My coursework average is {r['ca_avg']:.0f}% and my projected result is about {r['predicted']:.0f}%. "
            f"{reasons}\n\nI would like to meet you this week to go through the topics I am finding difficult and agree a "
            f"revision plan. Please let me know a time that suits you.\n\nThank you,\n{who}")
    return subject, body


def study_calendar_ics(plan: list[dict], start: _dt.date | None = None) -> str:
    """Turn the weekly study plan into a real calendar file (.ics) - 90 minute blocks, two slots per day."""
    start = start or (_dt.date.today() + _dt.timedelta(days=1))
    blocks = []
    for p in plan:                                     # plan is already sorted by need
        blocks += [p["Subject"]] * max(1, int(round(p["Hours / week"] / 1.5)))
    # interleave subjects so the same one is not studied twice in a row
    order, pool = [], list(blocks)
    while pool:
        pick = next((b for b in pool if not order or b != order[-1]), pool[0])
        pool.remove(pick); order.append(pick)
    slots = [(17, 0), (19, 0)]
    now = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//AI Academic Advisor//EN", "CALSCALE:GREGORIAN"]
    for i, subject in enumerate(order[:14]):
        day = start + _dt.timedelta(days=i // 2)
        h, m = slots[i % 2]
        s = _dt.datetime.combine(day, _dt.time(h, m))
        e = s + _dt.timedelta(minutes=90)
        lines += ["BEGIN:VEVENT", f"UID:study-{i}-{s:%Y%m%dT%H%M}@advisor", f"DTSTAMP:{now}",
                  f"DTSTART:{s:%Y%m%dT%H%M%S}", f"DTEND:{e:%Y%m%dT%H%M%S}",
                  f"SUMMARY:Study - {subject}", "DESCRIPTION:Planned by the AI Academic Advisor", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def agent_run(profile: dict, results: list) -> dict:
    """One full agent cycle. Returns the five stages plus the artefacts the agent produced."""
    if not results:
        return {}
    summ = overall_summary(results)
    pending = [r for r in results if r["status"] == "Predicted"]
    weak = sorted([r for r in results if r["weak"]], key=lambda r: -r["priority"])
    declining = [r for r in results if r["trend"] <= -8]
    hours = float(profile.get("study_hours") or 12)
    plan = study_plan(results, hours)

    perceive = [f"Read marks for {len(results)} subjects: {len(results) - len(pending)} with a final result, "
                f"{len(pending)} still awaiting an exam.",
                f"Student context: attendance {profile.get('attendance', 0):.0f}%, {hours:.0f} study hours a week."]
    analyse = [f"Projected average is {summ['average']:.0f}% ({summ['classification']}).",
               f"{len(weak)} of {len(results)} subjects are weak." +
               (f" Weakest: {weak[0]['subject']} at {weak[0]['predicted']:.0f}%." if weak else "")]
    for r in weak[:3]:
        analyse.append(f"{r['subject']}: {r['reasons'][0]}")
    if declining:
        analyse.append("Declining marks in: " + ", ".join(r["subject"] for r in declining) + ".")

    decide = []
    for r in weak:
        action = {"High": "escalate - lecturer consultation and a recovery plan",
                  "Medium": "intervene - focused practice and coursework repair"}[r["risk"]]
        decide.append(f"{r['subject']} ({r['risk']} risk): {action}.")
    if not weak:
        decide.append("No subject needs intervention; keep the current routine.")
    if plan and weak:
        decide.append(f"Moved the largest share of study time ({plan[0]['Hours / week']:.1f} h) to {plan[0]['Subject']}.")

    recommend = [f"{r['subject']}: {r['activities'][0]}" for r in weak[:3]] or \
                ["Stretch yourself with past-paper 'hard' questions."]

    emails = {r["subject"]: draft_consult_email(profile, r) for r in weak if r["risk"] == "High"}
    ics = study_calendar_ics(plan)
    act = [f"Built a weekly study timetable ({ics.count('BEGIN:VEVENT')} sessions) as a calendar file."]
    if emails:
        act.append("Drafted consultation emails to lecturers for: " + ", ".join(emails) + ".")
    if weak:
        act.append(f"Raised an early-warning flag for {len(weak)} subject(s).")
    return {"perceive": perceive, "analyse": analyse, "decide": decide, "recommend": recommend, "act": act,
            "emails": emails, "ics": ics, "plan": plan}


def model_report() -> dict:
    """Honest check of the prediction model on data it never saw (still synthetic - see _train_model)."""
    from sklearn.metrics import mean_absolute_error, r2_score
    model, _ = _train_model()
    rng = np.random.default_rng(2026)          # different seed from training
    n = 1000
    ca = np.clip(rng.normal(58, 15, n), 0, 100); trend = rng.normal(0, 8, n)
    att = rng.uniform(40, 100, n); study = rng.uniform(0, 28, n)
    y = np.clip(0.70 * ca + 0.35 * trend + 0.10 * (att - 75) + 0.30 * (np.minimum(study, 20) - 8) + 8
                + rng.normal(0, 6, n), 0, 100)
    pred = model.predict(np.column_stack([ca, trend, att, study]))
    return {"r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred)), "n": n}
