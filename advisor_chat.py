"""
advisor_chat.py - the Academic Advisor's conversation.

Understanding comes first, answering second:
  1. If a Gemini key is set, Gemini reads the question (with the student's unit names) and returns the intent and the unit as JSON.
  2. Otherwise a local classifier decides: TF-IDF word + character n-grams (typo tolerant) with Logistic Regression, trained on
     example questions for each intent. Nothing depends on a keyword being present.
  3. The unit the student means is found separately (a unit in their marks list, or ANY other unit name, which is researched).
  4. The answer is produced by the matching handler: calculations come from the agent's own model, resources come from research.
  5. If the question fits no intent, it is answered by research (Gemini, or Wikipedia) using the student's results as context.
"""
from __future__ import annotations

import re

import advisor_engine as eng

INTENT_EXAMPLES = {
    "weak": ["what are the units i am doing badly in", "which modules am i performing poorly in", "units where my marks are low", "which of my courses are the worst", "where am i falling behind", "what subjects should i be worried about", "which subjects am i weak in", "where am i struggling", "what are my worst units", "which units need my attention",
             "what should i focus on first", "my lowest subjects", "which courses am i failing", "what are my problem areas",
             "which units are dragging me down", "where do i need to improve most", "what are my weaknesses"],
    "strong": ["which module is my best", "units i am excelling in", "where do i score highest", "which are my strongest subjects", "what am i best at", "my top units", "where am i doing well", "my highest marks",
               "which subject do i perform best in", "what are my strengths", "which unit am i good at"],
    "predict": ["what grade am i likely to end up with", "what result should i expect at the end", "how will my final marks look", "what will i probably get", "can you estimate my end of semester grades", "where am i headed marks wise", "predict my final marks", "what will my final grade be", "forecast my results", "how will i perform in the exams",
                "what are my expected marks", "project my outcome", "what mark will i get", "estimate my final score",
                "what do you think i will score", "how will i end the semester"],
    "overall": ["what is my average across everything", "how am i doing in general", "am i on track overall", "what is my semester average", "my general performance", "what is my overall average", "what class of degree am i heading for", "my gpa", "how am i doing overall",
                "my overall standing", "what is my overall performance", "am i going to pass the semester", "what is my total average"],
    "resources": ["give me some good places to study X", "any websites for X", "what should i watch to learn X", "i want to learn more about X", "recommend something to read on X", "study guides for X", "where do i find past papers for X", "suggest learning resources for X", "recommend books and courses for X", "where can i learn X", "give me materials to study X",
                  "best resources for X", "youtube tutorials for X", "good websites to study X", "suggest best academic resourses for X",
                  "reading material for X", "what can i read to understand X", "any free courses on X", "links to learn X",
                  "which books should i use for X", "recommend resources", "i need study materials", "where can i find notes for X"],
    "improve": ["i am lost in X what now", "i do not understand X where do i start", "how can i get stronger in X", "what should i do to raise my mark in X", "where do i begin with X", "X is hard for me any advice", "how do i catch up in X", "how can i improve in X", "help me with X", "tips to do better in X", "how do i pass X", "i am struggling with X",
                "what should i do about X", "give me a plan for X", "how to revise for X", "how do i get better at X",
                "i keep failing X what do i do", "advice for X", "how should i study X"],
    "plan": ["make me a study plan", "how should i split my study time", "create a revision timetable", "schedule my study hours",
             "how many hours should i study", "weekly study plan please", "organise my week for studying", "build me a timetable"],
    "risk": ["am i likely to pass everything", "is there a chance i fail", "are any of my units in trouble", "do i have a chance of a retake", "am i safe or at risk", "which units am i at risk of failing", "will i fail anything", "do i need to retake any exam", "am i in danger of supplementary",
             "risk of failing", "could i repeat a unit", "am i going to fail", "which subjects might i fail"],
    "attendance": ["is skipping lectures hurting me", "should i go to more lectures", "how many classes have i attended", "am i attending enough", "how is my attendance", "does attendance affect my marks", "should i attend more classes", "my class attendance",
                   "is my attendance good enough"],
    "unit_info": ["how am i doing in X", "what is my mark in X", "tell me about my X performance", "status of X", "how is X going",
                  "what is my grade in X", "show me X results", "what are my X marks"],
    "greeting": ["good afternoon", "yo", "hello advisor", "hi there", "hello", "hi", "hey there", "good morning", "good evening", "hi advisor", "habari"],
    "thanks": ["cheers", "cool thanks", "nice one", "ok thanks", "perfect thank you", "awesome", "thank you", "thanks a lot", "great thanks", "i appreciate it", "that was helpful"],
    "other": ["what is photosynthesis", "explain recursion in simple words", "who is the president of kenya", "how does a neural network work",
              "define inflation", "what is the capital of france", "tell me a joke", "how do i cook ugali", "explain bubble sort",
              "what is the difference between tcp and udp", "how do plants make food", "what is machine learning", "who invented the telephone",
              "translate hello to swahili", "what is the meaning of life", "explain how a database index works", "what is the weather like"],
    "help": ["what questions can you answer", "how does this work", "what can you do", "help", "how do i use this", "what can i ask you", "what are your features", "show me what you can do"],
}
_UNITS = ["Entrepreneurship", "Marketing", "Calculus", "Database Systems", "Operating Systems", "Management Information System",
          "System Programming", "Economics", "Computer Networks", "Accounting"]
_NAMES = [n for n in INTENT_EXAMPLES if n != "other"]
_model = None


def _train():
    global _model
    if _model is not None:
        return _model
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    X, y = [], []
    for intent, exs in INTENT_EXAMPLES.items():
        for e in exs:
            if " X" in e or e.endswith(" X"):
                for u in _UNITS:                                     # the unit name must not decide the intent
                    X.append(e.replace("X", u.lower())); y.append(intent)
            else:
                X.append(e); y.append(intent)
    feats = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])
    _model = Pipeline([("f", feats), ("c", LogisticRegression(C=30, max_iter=3000))]).fit(X, y)
    return _model


def classify_intent(question: str, results=None) -> tuple[str, float]:
    """Local intent classifier. Unit names found in the question are masked so they cannot sway the decision."""
    q = question.lower()
    for r in results or []:
        q = q.replace(r["subject"].lower(), "unit")
    p = _train().predict_proba([q])[0]
    i = int(p.argmax())
    return str(_model.classes_[i]), float(p[i])


def evaluate() -> dict:
    """Accuracy on phrasings written after the training examples (reported honestly, not tuned to)."""
    held = [("which classes should i be most worried about", "weak"), ("tell me the unit where i shine", "strong"),
            ("roughly what will my semester results look like", "predict"), ("any good material to get better at calculus", "resources"),
            ("calculus is killing me, help", "improve"), ("split my week between subjects for studying", "plan"),
            ("is there a danger that i will fail a paper", "risk"), ("do i skip too many lectures", "attendance"),
            ("how is operating systems looking for me", "unit_info"), ("hey advisor", "greeting"), ("thank you so much", "thanks"),
            ("what can you help me with", "help"), ("explain what a hash table is", "other"), ("who won the world cup in 2018", "other"),
            ("in general how am i faring", "overall")]
    wrong = [(t, e, classify_intent(t)[0]) for t, e in held if classify_intent(t)[0] != e]
    return {"n": len(held), "accuracy": 1 - len(wrong) / len(held), "wrong": wrong}


# ----------------------------------------------------------------------------- finding the unit
_NOT_UNITS = re.compile(r"^(me|my|the|this|that|it|them|these|those|a|an|all|some|any|more|better|weak|worst|best|strong|priority|"
                        r"exam|exams|marks|grades|results|study|studying|learning|free|good|great|academic|online|"
                        r"unit|units|subject|subjects|course|courses|class|classes|semester|week|today|tomorrow)\b")


def extract_unit(question: str, results) -> tuple[dict | None, str]:
    """(unit from the student's marks, or None ; a unit name that is NOT in their marks, or '')."""
    q = question.lower().strip()
    for r in results:                                         # a listed unit named in the question
        if r["subject"].lower() in q:
            return r, ""
    phrases = re.findall(r"\b(?:for|in|on|about|of|with|at|into|learn|study|revise|understand|improve)\s+(?:the\s+)?"
                         r"(?:unit\s+|subject\s+|course\s+|module\s+)?([a-z0-9][a-z0-9 &+/\-']{2,60}?)"
                         r"(?:\s+(?:exam|exams|unit|subject|course|module|please|pls))?\s*[?.!]*$", q)
    for ph in reversed(phrases):
        ph = ph.strip()
        if ph and not _NOT_UNITS.match(ph):
            hit = eng._find_subject(ph, results)
            return (hit, "") if hit else (None, ph.title())
    hit = eng._find_subject(q, results)                       # typo tolerant match to a listed unit
    return (hit, "") if hit else (None, "")


# ----------------------------------------------------------------------------- Gemini as the reader
def gemini_understand(question: str, results, researcher):
    g = getattr(researcher, "gemini", None)
    if g is None or not g.available():
        return None
    units = ", ".join(r["subject"] for r in results)
    prompt = ("A university student is chatting with their academic advisor. Their units: " + units + ".\n"
              f"Student message: \"{question}\"\n"
              "Choose the intent: " + ", ".join(_NAMES) + ", other (anything else, including general study or subject questions).\n"
              "Also give the unit the message is about (a unit name exactly as the student wrote it, even if it is NOT in the list), or \"\" if none.\n"
              'Reply as JSON: {"intent": "...", "unit": "..."}')
    obj, _ = g.generate_json(prompt, "You classify student messages. Never answer them.", cache_key="adv:" + question.lower())
    if isinstance(obj, dict) and obj.get("intent") in _NAMES + ["other"]:
        return obj["intent"], str(obj.get("unit") or "").strip()
    return None


# ----------------------------------------------------------------------------- answers
def research_block(topic: str, researcher, note: str = "") -> str:
    r = eng.unit_research(topic, researcher)
    lines = [f"### {r['unit']}", note, r["summary"] or "", "**Resources**"]
    lines += [f"- [{x['title']}]({x['url']}) ({x['type']})" + (f" - {x['why']}" if x.get("why") else "") for x in r["resources"][:7]]
    if r["activities"]:
        lines += ["", "**Study activities**"] + [f"- {x}" for x in r["activities"][:4]]
    lines += ["", f"*{r['note']}*"]
    return "\n".join(l for l in lines if l is not None)


def _unit_status(r) -> str:
    return (f"**{r['subject']}**: predicted {r['predicted']:.0f}% (Grade {r['grade']}, {r['risk']} risk). "
            + (r["reasons"][0] if r["reasons"] else ""))


def respond(question: str, profile: dict, results: list, researcher=None) -> str:
    q = question.lower().strip()
    name = profile.get("name") or "there"
    if not results:
        return ("I don't have any marks to work with yet. Please enter your marks in the **Marks** tab "
                "(or load the sample data), then I can help. I can still research any unit if you ask, "
                "for example *resources for entrepreneurship*.") if not re.search(r"resourc|learn|study|book|course", q) else \
            research_block(extract_unit(q, [])[1] or question.title(), researcher)

    summ = eng.overall_summary(results)
    subj, other = extract_unit(question, results)

    # calculations that need a number in the question
    m = re.search(r"(?:score|get|got|obtain|make|scored)\s+(\d{1,3})\b", q)
    if m and subj and re.search(r"\bif\b|suppose|what happens", q):
        exam = min(100, int(m.group(1)))
        overall = eng.CA_WEIGHT * subj["ca_avg"] + eng.EXAM_WEIGHT * exam
        return (f"**What-if for {subj['subject']}:** with a coursework average of {subj['ca_avg']:.0f}% and an exam mark of {exam}%, "
                f"your final mark would be **{overall:.0f}% (Grade {eng.grade_of(overall)} - {eng.grade_label(overall)})**. "
                f"My model currently expects about {subj['predicted']:.0f}%.")
    m = re.search(r"(?:get|achieve|reach|aim for|target|score|earn|have|need)\s+(?:an?\s+)?(second upper|second lower|first|pass|[abcd]|\d{2,3})\b", q)
    if m and re.search(r"\bneed\b|to get|to reach|to achieve|aim|target|required", q):
        key = m.group(1)
        target = float(key) if key.isdigit() else eng.TARGETS.get(key)
        if target:
            lines = []
            for r in ([subj] if subj else results):
                need = eng.needed_exam(r["ca_avg"], target)
                if r["exam"] is not None:
                    lines.append(f"- {eng._fmt(r)}: exam already sat ({r['exam']:.0f}%).")
                elif need > 100:
                    lines.append(f"- **{r['subject']}**: needs {need:.0f}% in the exam, which is not reachable with current coursework.")
                elif need <= 0:
                    lines.append(f"- **{r['subject']}**: already secured. Any exam mark reaches {target:.0f}%.")
                else:
                    lines.append(f"- **{r['subject']}**: you need **{need:.0f}%** in the exam.")
            return f"To reach an overall **{target:.0f}%**:\n" + "\n".join(lines)

    # understand the question
    via = "local"
    understood = gemini_understand(question, results, researcher)
    if understood:
        intent, gem_unit = understood
        via = "gemini"
        if gem_unit and not subj:
            hit = eng._find_subject(gem_unit, results)
            subj, other = (hit, "") if hit else (None, gem_unit.title())
        conf = 1.0
    else:
        intent, conf = classify_intent(question, results)

    if intent == "other" or conf < 0.30:
        if researcher is not None:
            ans = researcher.answer_study_question(question, eng._summary_text(profile, results, summ))
            if ans and ans.get("answer"):
                src = "".join(f"\n- [{x['title']}]({x['url']})" for x in ans.get("sources", [])[:3])
                how = "Gemini with Google Search" if ans["mode"] == "gemini" else "Wikipedia"
                return ans["answer"] + (f"\n\n**Sources**{src}" if src else "") + f"\n\n*Researched with {how}.*"
        if subj:
            return _unit_status(subj) + f" Ask *'How can I improve {subj['subject']}?'* for a plan."
        return ("I did not understand that well enough to answer it reliably. You can ask about your weak or strong units, "
                "predictions, study plan, resources for any unit, or type **help**.")

    if intent in ("resources", "improve", "unit_info", "predict") and other:      # a unit that is not in the marks list
        if intent == "unit_info":
            return (f"**{other}** is not in your marks list, so I have no marks for it. Add it in the Marks tab and I can analyse it. "
                    f"Meanwhile ask me for *resources for {other}*.")
        return research_block(other, researcher, f"*{other} is not in your marks list, so this is researched from the unit name.*")

    if intent == "resources":
        targets = [subj] if subj else ([r for r in results if r["weak"]][:3] or [summ["worst"]])
        out = []
        for r in targets:
            out.append(research_block(r["subject"], researcher, _unit_status(r)))
        return "\n\n".join(out)

    if intent == "improve":
        if not subj:
            subj = sorted(results, key=lambda r: -r["priority"])[0]
            lead = f"You did not name a unit, so I picked your most urgent one.\n\n"
        else:
            lead = ""
        lines = [lead + f"### {subj['subject']} - {subj['predicted']:.0f}% (Grade {subj['grade']}, risk: {subj['risk']})", "**Why:**"]
        lines += [f"- {x}" for x in subj["reasons"]] + ["", "**What to do:**"] + [f"- {x}" for x in subj["activities"]]
        lines += ["", "**Resources:**"] + [f"- [{t}]({u})" for t, u in eng.resources_for(subj["subject"], researcher)]
        return "\n".join(lines)

    if intent == "unit_info":
        if not subj:
            return "Which unit do you mean? Name it, for example *How am I doing in " + results[0]["subject"] + "?*"
        r = subj
        ex = f"exam {r['exam']:.0f}%" if r["exam"] is not None else "exam not yet written"
        return (f"### {r['subject']}\nCoursework average {r['ca_avg']:.0f}%, {ex}. Predicted final **{r['predicted']:.0f}%** "
                f"(range {r['low']:.0f}-{r['high']:.0f}%), Grade {r['grade']}, risk **{r['risk']}**.\n\n" + "\n".join(f"- {x}" for x in r["reasons"]))

    if intent == "predict":
        if subj:
            return (f"For **{subj['subject']}** I predict **{subj['predicted']:.0f}%** (likely range {subj['low']:.0f}-{subj['high']:.0f}%, Grade {subj['grade']}).")
        lines = [f"- {eng._fmt(r)}  (range {r['low']:.0f}-{r['high']:.0f}%)" for r in results]
        return "Predicted final marks:\n" + "\n".join(lines) + f"\n\nOverall average **{summ['average']:.0f}%**, *{summ['classification']}*."

    if intent == "weak":
        weak = sorted([r for r in results if r["weak"]], key=lambda r: -r["priority"])
        if not weak:
            return f"Good news, {name}: none of your units is in the weak zone. Your lowest is {eng._fmt(summ['worst'])}."
        return ("These units need attention first (most urgent at the top):\n" +
                "\n".join(f"{i}. {eng._fmt(r)} - {r['reasons'][0]}" for i, r in enumerate(weak, 1)) +
                "\n\nAsk me *'How can I improve <unit>?'* for a tailored plan.")

    if intent == "strong":
        top = sorted(results, key=lambda r: -r["predicted"])[:3]
        return "Your strongest units:\n" + "\n".join(f"{i}. {eng._fmt(r)}" for i, r in enumerate(top, 1))

    if intent == "overall":
        return (f"Your projected overall average is **{summ['average']:.1f}%**, which is **{summ['classification']}**. "
                f"{summ['n_weak']} of {summ['n_subjects']} unit(s) are flagged as weak.")

    if intent == "plan":
        hrs = float(profile.get("study_hours") or 15)
        plan = eng.study_plan(results, hrs)
        return (f"Here is how to split your **{hrs:.0f} study hours** per week:\n" +
                "\n".join(f"- **{p['Subject']}**: {p['Hours / week']:.1f} h/week ({p['Priority']} priority)" for p in plan) +
                "\n\nStudy the hardest unit when you are freshest, in 50-minute blocks with 10-minute breaks.")

    if intent == "risk":
        risky = [r for r in results if r["risk"] in ("High", "Medium")]
        if not risky:
            return "No unit is currently at risk of failing. Keep it up!"
        return ("Units at risk:\n" + "\n".join(f"- {eng._fmt(r)} - risk **{r['risk']}**" for r in risky) +
                "\n\nAct early: see the lecturer and increase practice on these units.")

    if intent == "attendance":
        att = profile.get("attendance", 75)
        return f"Your recorded attendance is **{att:.0f}%**. " + ("Excellent." if att >= 85 else "Try to raise it above 85%; it noticeably improves marks.")

    if intent == "greeting":
        return (f"Hello {name}! I'm your academic advisor. Ask me anything in your own words: which units are weak, predictions, "
                "a study plan, or resources for any unit (even ones that are not in your marks list).")
    if intent == "thanks":
        return "You're welcome! Consistent effort beats last-minute cramming."
    return ("You can ask in your own words, for example:\n- *Which units am I weak in?*\n- *Suggest good resources for Entrepreneurship*\n"
            "- *How can I improve in Database Systems?*\n- *What do I need to get a B in Web Development?*\n- *Make me a study plan*\n"
            "- *What if I score 70 in the Computer Networks exam?*")
