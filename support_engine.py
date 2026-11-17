"""
support_engine.py
-----------------
NLP engine for the University Student Support Agent (Option 2).

Pipeline for every user message
  1. NLP pre-processing   - normalisation, synonym folding, entity extraction, sentiment cues
  2. Intent classification- TF-IDF (word + character n-grams) + Logistic Regression
  3. Information retrieval- TF-IDF cosine similarity over the knowledge base
  4. Decision             - blended score -> answer / clarify / fallback / human hand-off
  5. Context              - short follow-ups ("and the deadline?") reuse the previous topic
  6. Learning             - thumbs up/down and "I meant ..." corrections become new training
                            examples; the classifier is retrained instantly and the learning is saved.

NOTE: the knowledge base below is SAMPLE content so the agent can be demonstrated.
Replace the answers with your institution's official information before real use.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

INSTITUTION = "the University"

ANSWER_T = 0.42      # confident answer
CLARIFY_T = 0.24     # "did you mean ...?" between this and ANSWER_T
HUMAN_INTENT = "human_agent"
OOS_INTENT = "out_of_scope"
SMALLTALK = {"greeting", "goodbye", "thanks"}

# --------------------------------------------------------------------------
# Knowledge base (sample content)
# --------------------------------------------------------------------------
KB: list[dict] = [
    dict(id="greeting", title="Greeting", category="Small talk",
         answer="Hello! I am the student support assistant. I can help with fees, registration, exams, results, "
                "accommodation, the library, IT access, clearance and more. What do you need?",
         examples=["hello", "hi", "hi there", "good morning", "good afternoon", "good evening", "hey", "hello I need help",
                   "hi support", "habari", "greetings"],
         followups=["How do I pay my fees?", "When are the exams?"]),
    dict(id="goodbye", title="Goodbye", category="Small talk",
         answer="Goodbye, and good luck with your studies. Come back any time you need help.",
         examples=["bye", "goodbye", "see you later", "that is all", "i am done", "no more questions", "talk later",
                   "thats all for now", "cheers bye"],
         followups=[]),
    dict(id="thanks", title="Thanks", category="Small talk",
         answer="You are welcome. Is there anything else I can help you with?",
         examples=["thank you", "thanks", "thanks a lot", "i appreciate it", "that was helpful", "great thanks",
                   "many thanks", "awesome thank you", "cool thanks"],
         followups=[]),
    dict(id="human_agent", title="Speak to a support officer", category="Escalation",
         answer="I will hand this over to a support officer. I have logged a ticket with your conversation so you "
                "do not need to repeat yourself. A staff member will contact you through your student email.",
         examples=["i want to talk to a human", "speak to an agent", "connect me to a support officer",
                   "i need a real person", "escalate my issue", "i want to make a complaint", "call me please",
                   "transfer me to staff", "let me speak to someone", "human please"],
         followups=[]),
    dict(id="fee_payment", title="Paying fees", category="Finance",
         answer="**How to pay fees**\n- Pay through the bank or mobile-money channels published by the Finance Office.\n"
                "- Always use your **registration number** as the payment reference.\n"
                "- Keep the receipt or confirmation message until the payment shows on your statement.\n"
                "- Fees are normally due before semester registration opens.",
         examples=["how do i pay school fees", "fee payment methods", "where can i pay my tuition",
                   "can i pay fees using mpesa", "bank details for fees", "payment channels for fees",
                   "how to clear my fees", "what is the fee payment process", "pay tuition online",
                   "how do i pay my fee"],
         followups=["How do I check my fee balance?", "Are there bursaries or loans?"]),
    dict(id="fee_balance", title="Fee balance and statement", category="Finance",
         answer="**Checking your balance**\n- Log in to the student portal and open **Finance > Fee statement**.\n"
                "- Payments can take up to 48 hours to reflect.\n"
                "- If a payment is still missing after that, email the Finance Office with your registration number "
                "and a copy of the receipt.",
         examples=["how can i check my fee balance", "fee statement", "how much do i owe", "outstanding balance",
                   "statement of account", "my payment is not reflecting", "did my payment go through",
                   "i paid but the balance did not change", "view my fee arrears", "what is my balance"],
         followups=["How do I pay my fees?", "I want to talk to a human"]),
    dict(id="course_registration", title="Unit registration", category="Academics",
         answer="**Registering for units**\n- Open the student portal and choose **Registration > Semester registration**.\n"
                "- Select the units for your year and semester, then submit for approval.\n"
                "- You can add or drop units only during the dates in the academic calendar.\n"
                "- Late registration needs approval from your Head of Department and may attract a penalty.",
         examples=["how do i register for units", "semester registration", "when is unit registration",
                   "register my courses online", "how to add a unit", "drop a unit", "late registration",
                   "registration deadline", "enrol for the new semester", "course registration steps"],
         followups=["How do I check my fee balance?", "When are the exams?"]),
    dict(id="exam_timetable", title="Exam timetable and venues", category="Academics",
         answer="**Examinations**\n- The exam timetable is published on the student portal and notice boards about two weeks "
                "before exams start.\n- Bring your student ID and exam card; you need cleared fees to be issued a card.\n"
                "- Arrive at least 30 minutes early and check your venue on the timetable.",
         examples=["when are the exams", "exam timetable", "exam dates", "exam schedule for this semester",
                   "where is my exam venue", "what time does the exam start", "how do i get an exam card",
                   "when will the exam timetable be out", "final exams date", "what do i carry for exams"],
         followups=["What if I fail a unit?", "How do I check my results?"]),
    dict(id="class_timetable", title="Class timetable", category="Academics",
         answer="**Lecture timetable**\n- Class timetables are posted on the student portal and department notice boards "
                "at the start of the semester.\n- Changes are announced through your class representative and student email.",
         examples=["class timetable", "lecture schedule", "where is my lecture", "lecture venue",
                   "when do classes start", "timetable change", "what time is my lecture", "teaching timetable",
                   "my class schedule", "which room is the class"],
         followups=["When are the exams?"]),
    dict(id="results", title="Checking results", category="Academics",
         answer="**Results**\n- Provisional results are released on the student portal after the senate approves them.\n"
                "- Open **Academics > Results** and choose the semester.\n"
                "- If you think a mark is wrong, apply for a remark through your Head of Department within the stated period.",
         examples=["how do i check my results", "where can i see my grades", "when will results be released",
                   "view my exam results", "results portal", "i want to appeal my marks", "request a remark",
                   "my grade is wrong", "semester results", "see my marks online"],
         followups=["My marks are missing", "How do I get a transcript?"]),
    dict(id="missing_marks", title="Missing marks", category="Academics",
         answer="**Missing or incomplete marks**\n- Confirm you were registered for the unit and that fees were cleared.\n"
                "- Fill in the missing-marks form from your department and have your lecturer sign it.\n"
                "- Submit it to the Head of Department. Follow up after one week.",
         examples=["my marks are missing", "result not showing for one unit", "marks not posted", "incomplete result",
                   "missing marks for a subject", "my cat marks are not on the portal", "blank result on the portal",
                   "the lecturer has not uploaded my marks", "unit shows no grade", "mark not appearing"],
         followups=["How do I check my results?", "I want to talk to a human"]),
    dict(id="transcript", title="Transcripts and certificates", category="Academics",
         answer="**Transcripts**\n- Request an academic transcript from the Registrar's office or through the portal.\n"
                "- Fees must be fully cleared and a processing fee applies.\n"
                "- Official transcripts are issued in a sealed envelope; allow a few working days.",
         examples=["how do i request a transcript", "academic transcript", "official transcript", "result slip",
                   "i need a copy of my certificate", "transcript processing time", "apply for a transcript",
                   "get my transcript for a job application", "statement of results", "certified copy of my results"],
         followups=["How do I clear for graduation?"]),
    dict(id="supplementary", title="Supplementary and retake exams", category="Academics",
         answer="**Failed a unit?**\n- Failed units can be taken as **supplementary** or **retake** exams as the examination "
                "regulations allow.\n- Register for the exam in the period announced after results are released and pay "
                "the required fee.\n- Speak with your academic advisor about a revision plan.",
         examples=["what if i fail a unit", "supplementary exam", "retake an exam", "how do i resit a paper",
                   "special exam application", "repeat a unit", "i failed a course what next", "when are supplementary exams",
                   "carry forward a unit", "register for a retake"],
         followups=["When are the exams?", "How do I pay my fees?"]),
    dict(id="admission", title="Admission and applications", category="Admissions",
         answer="**Applying**\n- Check the programme's entry requirements on the admissions page.\n"
                "- Apply online during an intake window and attach certified copies of your certificates.\n"
                "- Admission letters are sent by email and posted on the portal; read the joining instructions carefully.",
         examples=["how do i apply to the university", "admission requirements", "how to join the university",
                   "application deadline", "when is the next intake", "cluster points needed", "i need an admission letter",
                   "new student joining instructions", "entry requirements for computer science", "undergraduate application"],
         followups=["Are there bursaries or loans?", "Where can I stay on campus?"]),
    dict(id="hostel", title="Accommodation", category="Student life",
         answer="**Accommodation**\n- Apply for a hostel room through the accommodation office or the portal once you are "
                "registered.\n- Rooms are allocated on a first-come basis; pay the residence fee to confirm your space.\n"
                "- Off-campus housing lists are available at the student welfare office.",
         examples=["how do i book a hostel", "accommodation on campus", "hostel fees", "room allocation",
                   "where can i stay", "off campus housing", "i need a room in the hostels", "apply for residence",
                   "hostel booking process", "student housing"],
         followups=["How do I pay my fees?"]),
    dict(id="library", title="Library services", category="Student life",
         answer="**Library**\n- The library opens on weekdays and Saturdays; extended hours apply during exams.\n"
                "- Use your student ID to borrow books and log in to e-resources and past papers.\n"
                "- Overdue books attract a fine; renew online before the due date.",
         examples=["library opening hours", "how do i borrow books", "library card", "access past papers",
                   "e-resources login", "fine for overdue books", "when does the library close",
                   "renew a book", "where can i find journals", "can i use the library on weekends"],
         followups=["I cannot log in to the portal"]),
    dict(id="ict_access", title="Portal, email and Wi-Fi access", category="IT",
         answer="**Login problems**\n- Use **Forgot password** on the portal login page and follow the reset email.\n"
                "- If you no longer receive email, visit the ICT help desk with your student ID.\n"
                "- Wi-Fi uses your portal credentials; update them on your device after a password change.",
         examples=["i forgot my password", "reset my portal password", "cannot log in to the student portal",
                   "e-learning access problem", "wifi password", "student email not working", "ict help desk",
                   "portal login issue", "my account is locked", "how do i access moodle"],
         followups=["How do I register for units?"]),
    dict(id="clearance", title="Graduation and clearance", category="Academics",
         answer="**Clearance and graduation**\n- Collect the clearance form from your department and get it signed by the library, "
                "finance, hostels and sports offices.\n- Graduation lists are published after senate approval.\n"
                "- Gowns are hired from the graduation office; certificates are collected after the ceremony.",
         examples=["when is graduation", "how do i clear for graduation", "clearance form", "graduation gown",
                   "final year clearance", "collect my certificate", "graduation list", "what is needed to graduate",
                   "clear from the library and finance", "graduation fees"],
         followups=["How do I get a transcript?"]),
    dict(id="deferment", title="Deferment, withdrawal and transfers", category="Academics",
         answer="**Deferring or withdrawing**\n- Write to the Dean of your school stating the reason and the period.\n"
                "- Submit supporting documents and get approval **before** the semester's cut-off date.\n"
                "- Changing course or transferring is handled by the Registrar and depends on entry requirements and space.",
         examples=["i want to defer my studies", "postpone the semester", "withdraw from the university",
                   "take a break from school", "change my course", "transfer to another programme",
                   "leave of absence", "how to freeze my studies", "drop out officially", "change of programme"],
         followups=["How do I pay my fees?"]),
    dict(id="financial_aid", title="Bursaries, loans and scholarships", category="Finance",
         answer="**Financial support**\n- Student loans and bursaries are applied for through the relevant funding bodies; "
                "keep your admission letter and ID handy.\n- The University also lists scholarships and sponsorship "
                "opportunities on the financial aid noticeboard.\n- Visit the financial aid office if your circumstances changed.",
         examples=["are there bursaries", "scholarship opportunities", "how do i apply for a student loan",
                   "financial aid", "sponsorship for students", "fee waiver", "i cannot afford fees",
                   "where do i apply for a bursary", "university scholarships", "help paying school fees"],
         followups=["How do I pay my fees?"]),
    dict(id="contacts", title="Offices and contacts", category="General",
         answer="**Finding an office**\n- The Registrar, Finance, Library and ICT offices have their contacts and opening hours "
                "on the University website.\n- Offices are normally open on weekdays during working hours.\n"
                "- For anything urgent, say 'talk to a human' and I will log a ticket.",
         examples=["where is the registrar's office", "office hours", "contact the finance office", "email address of the university",
                   "how can i reach the registrar", "where is the administration block", "working hours of offices",
                   "phone number for support", "how do i contact the department", "campus location"],
         followups=["I want to talk to a human"]),
    dict(id="student_services", title="Student ID, health and welfare", category="Student life",
         answer="**Student services**\n- Lost ID: report it to the student welfare office and apply for a replacement.\n"
                "- The campus clinic offers basic medical care; counselling services are confidential.\n"
                "- Clubs, sports and societies are listed at the student affairs office.",
         examples=["i lost my student id", "replace my student id card", "campus clinic", "medical services for students",
                   "student counselling", "how do i join a club", "sports activities", "student insurance",
                   "where is the health centre", "student welfare office"],
         followups=["Where is the registrar's office?"]),
    dict(id="out_of_scope", title="Out of scope", category="Fallback",
         answer="That is outside what I can help with. I support questions about studying at the University.",
         examples=["what is the weather today", "tell me a joke", "who won the football match", "bitcoin price",
                   "how do i cook ugali", "recommend a movie", "what is the capital of france", "sing a song",
                   "best phone to buy", "latest news", "what is your favourite colour", "write me a poem"],
         followups=[]),
]

EXTRA_EXAMPLES = {
    "fee_payment": ["fees", "pay fees", "tuition payment", "how to pay"],
    "fee_balance": ["what do i owe", "how much money do i owe the university", "amount i still need to pay",
                    "my deposit is not appearing on my account", "payment not showing on my statement",
                    "i sent money but it is not on my balance", "balance", "fee arrears"],
    "course_registration": ["registration", "register units", "sign up for my units", "enrol"],
    "exam_timetable": ["exams", "exam timetable", "exam dates"],
    "missing_marks": ["i cannot see my score for one subject", "my score is not showing", "score for a unit is missing",
                      "missing marks", "no mark for a unit"],
    "results": ["results", "my grades", "marks"],
    "transcript": ["transcript", "transcripts"],
    "supplementary": ["supplementary", "retake", "i failed a unit", "failed math what now", "resit"],
    "admission": ["how do i get admitted", "how to get admission", "how can i be admitted", "admission", "apply",
                  "i want to study here", "joining the university"],
    "hostel": ["hostel", "accommodation", "is there a place to sleep on campus", "rooms for students"],
    "library": ["library", "books", "library hours", "what time does the library close"],
    "ict_access": ["password", "forgot login", "wifi", "portal not opening", "i forgot my login details"],
    "clearance": ["graduation", "clearance", "graduation ceremony date"],
    "deferment": ["defer", "withdraw", "pause my studies", "deferment"],
    "financial_aid": ["bursary", "scholarship", "loan", "sponsorship"],
    "contacts": ["contacts", "registrar", "office hours", "where is the office"],
    "student_services": ["student id", "clinic", "counselling", "lost id"],
    "class_timetable": ["class timetable", "lectures", "timetable"],
    "human_agent": ["human", "agent", "speak to a person", "i want to speak to a person", "customer care"],
}
for _a in KB:
    _a["examples"] = _a["examples"] + EXTRA_EXAMPLES.get(_a["id"], [])

SYNONYMS = {
    "m-pesa": "mpesa", "mpesa": "mpesa", "fees": "fee", "tuition": "fee", "pwd": "password", "uni": "university",
    "cats": "cat", "exams": "exam", "marks": "marks", "grades": "marks", "grade": "marks", "hostels": "hostel",
    "dorm": "hostel", "dormitory": "hostel", "login": "log in", "signin": "log in",
}
FRUSTRATION = {"useless", "frustrated", "angry", "terrible", "worst", "waste", "annoyed", "ridiculous", "stupid",
               "nonsense", "unacceptable", "fed up", "not helping", "still not", "again and again", "pathetic"}
REG_RE = re.compile(r"\b[A-Za-z]{1,4}\d{1,4}/\d{2,6}(?:/\d{2,4})?\b")
MONEY_RE = re.compile(r"\b(?:kes|ksh|kshs)\.?\s?[\d,]+(?:\.\d+)?\b|\b[\d,]{3,}\s?(?:kes|ksh|kshs)\b", re.I)
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)


def normalise(text: str) -> str:
    t = text.lower().replace("can't", "cannot").replace("won't", "will not").replace("don't", "do not")
    t = re.sub(r"[^a-z0-9/\s\-']", " ", t)
    words = [SYNONYMS.get(w, w) for w in t.split()]
    return " ".join(words)


def extract_entities(text: str) -> dict:
    ents = {}
    if m := REG_RE.findall(text):
        ents["registration_number"] = m[0]
    if m := MONEY_RE.findall(text):
        ents["amount"] = m[0].strip()
    if m := DATE_RE.findall(text):
        ents["date"] = m[0]
    return ents


def is_frustrated(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in FRUSTRATION) or ("!!" in text)


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
class SupportBot:
    def __init__(self, learned_path: str | Path | None = None):
        self.kb: list[dict] = [dict(a) for a in KB]
        self.extra: list[tuple[str, str]] = []          # learned (text, intent)
        self.penalty: dict[str, float] = {}
        self.log: list[dict] = []
        self.tickets: list[dict] = []
        self.path = Path(learned_path) if learned_path else None
        self._load()
        self.train()

    # ---------------- persistence of what the agent has learned
    def _load(self):
        if not self.path or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text())
            self.extra = [tuple(x) for x in d.get("extra", [])]
            self.penalty = d.get("penalty", {})
            for a in d.get("custom_articles", []):
                if all(a["id"] != k["id"] for k in self.kb):
                    self.kb.append(a)
        except Exception:
            pass

    def save(self):
        if not self.path:
            return
        try:
            custom = [a for a in self.kb if a.get("custom")]
            self.path.write_text(json.dumps({"extra": self.extra, "penalty": self.penalty,
                                             "custom_articles": custom}, indent=1))
        except Exception:
            pass

    # ---------------- training
    def _training_data(self):
        X, y = [], []
        for a in self.kb:
            for ex in a["examples"] + [a["title"]]:
                X.append(normalise(ex)); y.append(a["id"])
        for t, i in self.extra:
            X.append(normalise(t)); y.append(i)
        return X, y

    def _pipeline(self):
        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        return Pipeline([("f", feats), ("c", LogisticRegression(C=40, max_iter=2000))])

    def train(self):
        X, y = self._training_data()
        self.clf = self._pipeline().fit(X, y)
        self.classes = list(self.clf.classes_)
        docs = [normalise(" ".join([a["title"], " ".join(a["examples"]), a["answer"]])) for a in self.kb]
        self.ret = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(docs)
        self.doc_matrix = self.ret.transform(docs)
        self.ids = [a["id"] for a in self.kb]

    def evaluate(self) -> dict:
        """Cross-validated accuracy of the intent classifier on the training utterances."""
        X, y = self._training_data()
        counts = pd.Series(y).value_counts()
        k = int(min(5, counts.min()))
        if k < 2:
            return {"accuracy": None, "n": len(X), "intents": len(counts)}
        scores = cross_val_score(self._pipeline(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=1))
        return {"accuracy": float(scores.mean()), "n": len(X), "intents": int(len(counts))}

    # ---------------- scoring
    def score(self, text: str) -> list[tuple[str, float, float, float]]:
        """Return [(intent, blended, p_intent, cosine)] sorted best first."""
        q = normalise(text)
        probs = dict(zip(self.classes, self.clf.predict_proba([q])[0]))
        sims = linear_kernel(self.ret.transform([q]), self.doc_matrix)[0]
        sim_by = dict(zip(self.ids, sims))
        out = []
        for i in self.ids:
            p, c = probs.get(i, 0.0), float(sim_by[i])
            s = 0.6 * p + 0.4 * min(1.0, c * 2.0) - self.penalty.get(i, 0.0)
            out.append((i, s, p, c))
        return sorted(out, key=lambda r: -r[1])

    def article(self, intent: str) -> dict:
        return next(a for a in self.kb if a["id"] == intent)

    # ---------------- main entry point
    def respond(self, text: str, history: list[dict] | None = None) -> dict:
        history = history or []
        ranked = self.score(text)
        context_used = False
        effective = text

        # short follow-up with weak signal -> blend with the previous question
        clearly_off_topic = ranked[0][0] == OOS_INTENT and ranked[0][1] >= 0.15
        if ranked[0][1] < ANSWER_T and len(text.split()) <= 4 and not clearly_off_topic:
            prev = next((m for m in reversed(history) if m["role"] == "user"), None)
            if prev:
                combo = prev["text"] + " " + text
                r2 = self.score(combo)
                if (r2[0][1] > ranked[0][1] + 0.05 and r2[0][1] >= CLARIFY_T
                        and r2[0][0] not in SMALLTALK and r2[0][0] != OOS_INTENT):
                    ranked, effective, context_used = r2, combo, True

        top_id, top_score, p, c = ranked[0]
        art = self.article(top_id)
        frustrated = is_frustrated(text)
        cands = [{"intent": i, "title": self.article(i)["title"], "score": round(s, 3)}
                 for i, s, _, _ in ranked if i not in SMALLTALK and i != OOS_INTENT][:3]

        if top_id == OOS_INTENT:
            status = "fallback"
        elif top_id == HUMAN_INTENT:
            status = "escalate"
        elif top_id in SMALLTALK and top_score >= CLARIFY_T:
            status = "smalltalk"
        elif top_score >= ANSWER_T:
            status = "answered"
        elif top_score >= CLARIFY_T:
            status = "clarify"
        else:
            status = "fallback"

        reply, followups = "", []
        if status in ("answered", "smalltalk"):
            reply, followups = art["answer"], art.get("followups", [])
        elif status == "escalate":
            reply = art["answer"]
        elif status == "clarify":
            reply = "I am not completely sure what you mean. Did you want help with one of these?"
        else:
            reply = ("I could not find a reliable answer to that in my knowledge base, so I would rather not guess. "
                     "Try rephrasing with a keyword (for example *fees*, *registration*, *exam*), pick a topic below, "
                     "or ask for a human.")
        if frustrated and status != "escalate":
            reply = "I am sorry for the trouble. " + reply
            if status in ("fallback", "clarify"):
                reply += "\n\nIf you prefer, type **human** and I will pass this to a support officer."

        ticket = None
        if status == "escalate" or (frustrated and status == "fallback"):
            ticket = self._ticket(text, history)
            reply += f"\n\nTicket **{ticket['id']}** has been logged (simulated in this demo)."

        entry = {"ts": time.time(), "text": text, "intent": top_id, "confidence": round(top_score, 3),
                 "status": status, "feedback": 0, "context_used": context_used}
        self.log.append(entry)
        return {"text": reply, "intent": top_id, "title": art["title"], "confidence": top_score, "status": status,
                "candidates": cands, "entities": extract_entities(text), "sentiment": "frustrated" if frustrated else "neutral",
                "context_used": context_used, "followups": followups, "query": effective,
                "retrieval": c, "classifier": p, "ticket": ticket, "log_index": len(self.log) - 1}

    def answer_for(self, intent: str) -> dict:
        a = self.article(intent)
        return {"text": a["answer"], "intent": intent, "title": a["title"], "followups": a.get("followups", [])}

    def _ticket(self, text, history):
        t = {"id": f"TKT-{len(self.tickets) + 1:04d}", "question": text,
             "transcript": [f"{m['role']}: {m['text']}" for m in history[-6:]]}
        self.tickets.append(t)
        return t

    # ---------------- learning from feedback
    def feedback(self, log_index: int, helpful: bool):
        """Thumbs up/down on an answer. Up = learn the phrasing, down = lower that topic's score."""
        e = self.log[log_index]
        e["feedback"] = 1 if helpful else -1
        i = e["intent"]
        if helpful:
            if e["status"] == "answered" and (e["text"].lower(), i) not in self.extra:
                self.extra.append((e["text"], i))
                self.train()
            self.penalty[i] = max(0.0, self.penalty.get(i, 0.0) - 0.02)
        elif e["status"] in ("answered", "clarify"):
            self.penalty[i] = min(0.2, self.penalty.get(i, 0.0) + 0.04)
        self.save()

    def correct(self, log_index: int, right_intent: str):
        """The user tells the agent what they really meant -> new labelled training example."""
        e = self.log[log_index]
        self.extra.append((e["text"], right_intent))
        e["feedback"] = 1
        e["intent_corrected"] = right_intent
        self.penalty[right_intent] = max(0.0, self.penalty.get(right_intent, 0.0) - 0.04)
        self.train()
        self.save()

    def add_article(self, title: str, answer: str, examples: list[str], category: str = "Custom") -> str:
        base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "custom"
        aid, n = base, 2
        while any(a["id"] == aid for a in self.kb):
            aid, n = f"{base}_{n}", n + 1
        self.kb.insert(len(self.kb) - 1, dict(id=aid, title=title, category=category, answer=answer,
                                              examples=examples, followups=[], custom=True))
        self.train()
        self.save()
        return aid

    # ---------------- analytics
    def stats(self) -> dict:
        df = pd.DataFrame(self.log)
        if df.empty:
            return {"df": df, "n": 0}
        rated = df[df["feedback"] != 0]
        return {"df": df, "n": len(df),
                "answered": int((df["status"].isin(["answered", "smalltalk"])).sum()),
                "fallback": int((df["status"] == "fallback").sum()),
                "satisfaction": float((rated["feedback"] == 1).mean()) if len(rated) else None,
                "avg_conf": float(df["confidence"].mean())}
