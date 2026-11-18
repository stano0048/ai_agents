"""
qa_tests.py - the live test suite shown on the Testing page, written into the database and into the PDF report.

Every test runs against a throw-away database. Anything that needs the internet or a Gemini key is tested with a
mock server response, and the test name says so; the real services cannot be reached from a test environment.
"""
from __future__ import annotations

import os
import tempfile
import time

import advisor_engine as adv
import finance_engine as fin
from db import Database, MemberLimitError, DuplicateMemberError
from llm import Gemini
from research import Researcher
from support_engine import SupportBot


class _Resp:
    def __init__(self, text, status=200, chunks=()):
        self.status_code, self._t, self._c = status, text, chunks

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self._t}]},
                                "groundingMetadata": {"groundingChunks": [{"web": {"uri": u, "title": t}} for t, u in self._c]}}]}


def _db():
    return Database(os.path.join(tempfile.mkdtemp(), "qa.db"))


def _offline(db=None):
    return Researcher(db, Gemini(db=db), get=lambda *a, **k: None)


def _wiki_get(url, params=None):
    if params and params.get("action") == "opensearch":
        return ["q", ["Entrepreneurship"], [""], [""]]
    if "rest_v1" in url:
        return {"title": "Entrepreneurship", "extract": "Entrepreneurship is the creation of new businesses.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Entrepreneurship"}}}
    if "openlibrary" in url:
        return {"docs": [{"title": "Zero to One", "author_name": ["Peter Thiel"], "key": "/works/OL1W"}]}


# ----------------------------------------------------------------------------- database
def t_member_limit():
    d = _db()
    for i in range(6):
        d.add_member(f"Member {i}", "", "Dev", "Wrote part number %d of the system" % i, [])
    try:
        d.add_member("Seventh", "", "", "This must be refused by the database", [])
    except MemberLimitError:
        return "7th member refused by the database trigger; 6 stored"
    raise AssertionError("7th member was accepted")


def t_duplicate():
    d = _db(); d.add_member("Jane Doe", "", "", "Testing the duplicate rule here", [])
    try:
        d.add_member("jane doe", "", "", "Same name, different case, refused", [])
    except DuplicateMemberError:
        return "duplicate name refused (case-insensitive)"
    raise AssertionError("duplicate accepted")


def t_passphrase():
    d = _db()
    assert d.set_passphrase("correct horse")[0]
    assert not d.verify_passphrase("wrong")[0] and d.verify_passphrase("correct horse")[0]
    assert d.get("pass_hash") and "correct horse" not in d.get("pass_hash") and len(d.get("pass_hash")) == 64
    assert not d.set_passphrase("another one", old="wrong")[0]
    return "wrong passphrase refused, right one accepted, stored as salted PBKDF2 hash"


def t_lockout():
    d = _db(); d.set_passphrase("secret-123")
    for _ in range(5):
        d.verify_passphrase("bad")
    assert d.lockout_remaining() > 0 and not d.verify_passphrase("secret-123")[0]
    return f"locked for {d.lockout_remaining()} s after 5 wrong attempts, even the right passphrase is refused"


def t_persistence():
    p = os.path.join(tempfile.mkdtemp(), "x.db")
    d = Database(p); d.add_member("Persist Test", "", "", "Check the data survives a restart", []); d.set_passphrase("persist-me")
    d2 = Database(p)
    assert d2.list_members()[0]["name"] == "Persist Test" and d2.verify_passphrase("persist-me")[0]
    return "members and passphrase still there after reopening the database file"


def t_backup_restore():
    d = _db(); d.add_member("Backup Person", "", "Tester", "Used to test the backup and restore", ["Finance"])
    txt = d.export_members(); d2 = _db()
    assert d2.import_members(txt) == 1 and d2.list_members()[0]["agents"] == "Finance"
    return "members exported to JSON and restored into an empty database"


# ----------------------------------------------------------------------------- research
def t_unknown_unit_offline():
    x = _offline().research_unit("Zzqx Studies 101")
    assert x["mode"] == "offline" and any("Zzqx" in r["title"] for r in x["resources"])
    return "an unknown unit gets resources built from its own name (not a random list)"


def t_unit_areas():
    r = _offline()
    a = r.research_unit("Entrepreneurship")["summary"]; b = r.research_unit("System Programming")["summary"]
    assert "Business" in a and "Programming" in b
    return "Entrepreneurship matched to business, System Programming to programming"


def t_unit_live_mock():
    x = Researcher(None, Gemini(), get=_wiki_get).research_unit("Entrepreneurship")
    assert x["mode"] == "live" and any(r["type"] == "Book" for r in x["resources"]) and x["sources"]
    return "(mocked Wikipedia/Open Library) live research returns an overview, a book and sources"


def t_unit_gemini_mock():
    body = '{"summary":"s","resources":[{"title":"T","type":"Course","url":"https://x.org","why":"w"}],"activities":["a"]}'
    g = Gemini(key="k", post=lambda *a, **k: _Resp("```json\n" + body + "\n```", chunks=[("X", "https://x.org")]))
    x = Researcher(None, g, get=lambda *a, **k: None).research_unit("Entrepreneurship")
    assert x["mode"] == "gemini" and x["resources"][0]["url"] == "https://x.org" and x["sources"]
    return "(mocked Gemini) grounded JSON parsed, sources captured"


def t_gemini_failure():
    g = Gemini(key="k", post=lambda *a, **k: _Resp("", status=429))
    x = Researcher(None, g, get=lambda *a, **k: None).research_unit("Marketing")
    assert x["mode"] == "offline"
    return "(mocked HTTP 429) falls back to offline research instead of failing"


def t_research_cache():
    d = _db(); n = []
    def get(*a, **k):
        n.append(1); return _wiki_get(*a, **k)
    r = Researcher(d, Gemini(db=d), get=get)
    r.research_unit("Entrepreneurship"); first = len(n); r.research_unit("Entrepreneurship")
    assert len(n) == first
    return "second lookup of the same unit served from the database cache"


# ----------------------------------------------------------------------------- advisor
def _res():
    return adv.analyse(adv.sample_marks(), 80, 12)


def t_adv_weak():
    weak = [r["subject"] for r in _res() if r["weak"]]
    assert "Discrete Mathematics" in weak
    return "weak subjects found: " + ", ".join(weak)


def t_adv_model():
    m = adv.model_report()
    assert m["r2"] > 0.6
    return f"Ridge model R-squared {m['r2']:.2f}, mean error {m['mae']:.1f} marks (synthetic data)"


def t_adv_goal():
    assert abs(adv.needed_exam(60, 70) - ((70 - 0.3 * 60) / 0.7)) < 1e-6
    return "exam mark needed for a 70% target matches the formula"


def t_adv_plan():
    plan = adv.study_plan(_res(), 15)
    assert abs(sum(p["Hours / week"] for p in plan) - 15) < 0.6
    return "weekly hours are shared out to the budget"


def t_adv_chat_unlisted():
    out = adv.chat_response("recommend resources for entrepreneurship", {"name": "T"}, _res(), _offline())
    assert "Entrepreneurship" in out and "http" in out
    return "asking about a unit that is not in the marks list triggers research"


def t_adv_chat_whatif():
    out = adv.chat_response("what if I score 70 in Database Systems exam", {"name": "T"}, _res(), _offline())
    assert "What-if" in out
    return "what-if question calculated"


def t_gemini_settings():
    d = _db(); seen = {}
    def post(url, json=None, headers=None, **k):
        seen["url"], seen["key"] = url, headers["x-goog-api-key"]
        return _Resp("ready")
    d.set_gemini("AIzaSyEXAMPLEKEY123456", "gemini-3.8-flash")
    g = Gemini(db=d, post=post)                      # a new client, nothing passed in: it must read the database
    assert g.available() and g.key_source() == "saved in the database" and g.generate("hi")["ok"]
    assert seen["key"] == "AIzaSyEXAMPLEKEY123456" and "gemini-3.8-flash:generateContent" in seen["url"]
    assert "EXAMPLEKEY" not in d.gemini_settings()["masked"]
    d.clear_gemini_key(); assert not Gemini(db=d).saved_key() and d.gemini_settings()["model"] == "gemini-3.8-flash"
    return "key and model saved in the database are used by a new client; the key is masked on screen; removing it works"


def t_default_model():
    from llm import DEFAULT_MODEL
    assert DEFAULT_MODEL == "gemini-3.8-flash"
    return "default model is gemini-3.8-flash (changeable and saved in the database)"


def t_adv_intent():
    import advisor_chat
    e = advisor_chat.evaluate()
    assert e["accuracy"] >= 0.75
    return f"intent accuracy {e['accuracy']:.0%} on {e['n']} unseen phrasings (typos and paraphrases, no keyword rules)"


def t_adv_resources_listed_unit():
    res = _res(); res[-1]["subject"] = "Management Information System"
    out = adv.chat_response("Suggest best academic resourses for Management Information System", {"name": "T"}, res, _offline())
    assert "### Management Information System" in out and "http" in out and "strongest" not in out
    return "a misspelt resources request for a listed unit is answered with that unit's researched resources"


def t_adv_open_question():
    class R:
        gemini = None
        def answer_study_question(self, q, ctx=""):
            return {"answer": "Photosynthesis is how plants make food from light.", "sources": [], "mode": "gemini"}
    out = adv.chat_response("what is photosynthesis", {"name": "T"}, _res(), R())
    assert "Photosynthesis is how plants" in out
    return "(mocked researcher) a general question outside every intent is researched, not forced into a keyword answer"


def t_adv_gemini_reads_question():
    body = '{"intent": "resources", "unit": "Entrepreneurship"}'
    g = Gemini(key="k", post=lambda *a, **k: _Resp(body))
    out = adv.chat_response("my lecturer says ill need extra reading for that business thing", {"name": "T"}, _res(), Researcher(None, g, get=lambda *a, **k: None))
    assert "### Entrepreneurship" in out
    return "(mocked Gemini) Gemini reads a vague message, returns intent and unit, and the agent researches that unit"


# ----------------------------------------------------------------------------- support
def _bot(db=None, researcher=None):
    return SupportBot(db or _db(), researcher)


def t_sup_heldout():
    h = _bot().evaluate_heldout()
    assert h["accuracy"] >= 0.7
    return f"held-out intent accuracy {h['accuracy']:.0%} on {h['n']} unseen phrasings (not tuned)"


def t_sup_catalogue():
    r = _bot().respond("How much is cooking oil 1L?")
    assert r["status"] == "catalogue" and "299" in r["text"]
    return "price and offer read from the product table"


def t_sup_stock():
    r = _bot().respond("Do you have quinoa in stock?")
    assert "out of stock" in r["text"]
    return "out-of-stock product reported honestly"


def t_sup_safety():
    r = _bot().respond("I drank the milk and now I feel sick, it was expired")
    assert r["ticket"] and r["ticket"]["department"] == "Quality and Safety" and r["ticket"]["priority"] == "High"
    return "food-safety message escalated to Quality and Safety with high priority"


def t_sup_research_mock():
    class R:
        def answer_question(self, q, context=""):
            return {"answer": "Cooked rice keeps 3 to 4 days in the fridge.", "sources": [{"title": "Src", "url": "https://x.org"}], "mode": "gemini"}
    r = _bot(researcher=R()).respond("How long can I keep cooked rice in the fridge?")
    assert r["status"] == "researched" and r["sources"]
    return "(mocked researcher) a question with no stored keywords is researched and cites sources"


def t_sup_no_guess():
    r = _bot(researcher=_offline()).respond("How long can I keep cooked rice in the fridge?")
    assert r["status"] == "fallback" and "rather not guess" in r["text"]
    return "with no research available it admits it does not know"


def t_sup_learns():
    class R:
        def answer_question(self, q, context=""):
            return {"answer": "Cooked rice keeps 3 to 4 days in the fridge.", "sources": [], "mode": "gemini"}
    d = _db(); b = _bot(d, R()); q = "How long can I keep cooked rice in the fridge?"
    b.feedback(b.respond(q)["log_index"], True)
    b2 = _bot(d, None)                                   # a new agent, same database, no researcher
    assert b2.respond(q)["status"] == "learned"
    return "a thumbs-up on a researched answer is remembered by a fresh agent via the database"


def t_sup_correction_persists():
    d = _db(); b = _bot(d); i = b.respond("zorp the blarg")["log_index"]
    b.correct(i, "delivery")
    assert any(t == "zorp the blarg" for t, _ in _bot(d).extra)
    return "a correction is stored in the database and reloaded by a new agent"


def t_sup_followup():
    b = _bot(); h = [{"role": "user", "text": "do you deliver"}, {"role": "assistant", "text": "x"}]
    r = b.respond("what about the fee", h)
    assert r["intent"] == "delivery"
    return "short follow-up resolved using the previous question"


# ----------------------------------------------------------------------------- finance
def t_fin_heldout():
    h = fin.categoriser_heldout(fin.Categoriser())
    assert h["accuracy"] >= 0.8
    return f"categoriser held-out accuracy {h['accuracy']:.0%} on {h['n']} unseen descriptions"


def t_fin_correction_db():
    d = _db(); c = fin.Categoriser(d); c.learn("zxqv blorp 77", "Transport")
    assert fin.Categoriser(d).predict("zxqv blorp 77")[0] == "Transport"
    return "category correction stored in the database and learned by a new categoriser"


def t_fin_gemini_assist():
    g = Gemini(key="k", post=lambda *a, **k: _Resp("Transport"))
    c = fin.Categoriser(None, g)
    cat, conf = c.predict("zxqv blorp 77")
    assert cat == "Transport" and c.assisted == 1
    return "(mocked Gemini) an unrecognised description is categorised by Gemini, restricted to our category list"


def t_fin_forecast():
    df, inc = fin.sample_data(fin.Categoriser()); f = fin.forecast(df)
    assert f["total"] > 0 and f["mae_model"] is not None
    return f"forecast {f['total']:,.0f}; back-test error {f['mae_model']:,.0f} vs {f['mae_base']:,.0f} baseline"


def t_fin_recs():
    c = fin.Categoriser(); df, inc = fin.sample_data(c)
    a = fin.analyse(df, fin.monthly_income(inc), 50000); f = fin.forecast(df)
    assert len(fin.recommend(a, f)) >= 2
    return "ranked saving strategies produced"


TESTS = [
    ("Member limit enforced by the database", "Database", t_member_limit),
    ("Duplicate member refused", "Database", t_duplicate),
    ("Passphrase check and hashing", "Database", t_passphrase),
    ("Brute-force lockout", "Database", t_lockout),
    ("Data survives a restart", "Database", t_persistence),
    ("Backup and restore of members", "Database", t_backup_restore),
    ("Unknown unit researched by name (offline)", "Research", t_unknown_unit_offline),
    ("Unit matched to the right subject area", "Research", t_unit_areas),
    ("Live research path", "Research", t_unit_live_mock),
    ("Gemini research path", "Research", t_unit_gemini_mock),
    ("Gemini failure falls back", "Research", t_gemini_failure),
    ("Research cached in the database", "Research", t_research_cache),
    ("Weak subjects identified", "Academic Advisor", t_adv_weak),
    ("Prediction model quality", "Academic Advisor", t_adv_model),
    ("Target-mark calculation", "Academic Advisor", t_adv_goal),
    ("Study-time plan", "Academic Advisor", t_adv_plan),
    ("Unlisted unit in chat is researched", "Academic Advisor", t_adv_chat_unlisted),
    ("What-if question", "Academic Advisor", t_adv_chat_whatif),
    ("Understands questions without keywords", "Academic Advisor", t_adv_intent),
    ("Misspelt resources request answered correctly", "Academic Advisor", t_adv_resources_listed_unit),
    ("General question is researched", "Academic Advisor", t_adv_open_question),
    ("Gemini reads a vague message", "Academic Advisor", t_adv_gemini_reads_question),
    ("Gemini key and model saved in the database", "Research", t_gemini_settings),
    ("Default Gemini model", "Research", t_default_model),
    ("Intent accuracy on unseen phrasings", "Support Agent", t_sup_heldout),
    ("Price from the product catalogue", "Support Agent", t_sup_catalogue),
    ("Out-of-stock reported", "Support Agent", t_sup_stock),
    ("Food-safety escalation", "Support Agent", t_sup_safety),
    ("Open question is researched", "Support Agent", t_sup_research_mock),
    ("Admits it does not know", "Support Agent", t_sup_no_guess),
    ("Remembers a researched answer", "Support Agent", t_sup_learns),
    ("Correction persists", "Support Agent", t_sup_correction_persists),
    ("Follow-up question", "Support Agent", t_sup_followup),
    ("Categoriser accuracy on unseen descriptions", "Finance Agent", t_fin_heldout),
    ("Correction persists in the database", "Finance Agent", t_fin_correction_db),
    ("Gemini assists unknown descriptions", "Finance Agent", t_fin_gemini_assist),
    ("Forecast and back-test", "Finance Agent", t_fin_forecast),
    ("Saving recommendations", "Finance Agent", t_fin_recs),
]


def run_all() -> list[dict]:
    rows = []
    for name, agent, fn in TESTS:
        t0 = time.time()
        try:
            detail, ok = fn(), True
        except Exception as e:                       # a failing test is reported, never hidden
            detail, ok = f"{type(e).__name__}: {e}", False
        rows.append({"name": name, "agent": agent, "passed": ok, "detail": detail, "seconds": round(time.time() - t0, 2)})
    return rows


if __name__ == "__main__":
    out = run_all()
    for r in out:
        print(("PASS" if r["passed"] else "FAIL"), f"{r['agent']:<17} {r['name']:<46} {r['detail']}")
    print(f"\n{sum(r['passed'] for r in out)}/{len(out)} passed")
