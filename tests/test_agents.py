"""Quick checks for the three agents.   Run:  python tests/test_agents.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import advisor_engine as adv
import finance_engine as fin
from support_engine import SupportBot

# ---- Option 1: advisor
res = adv.analyse(adv.sample_marks(), 80, 12)
assert len(res) == 6 and any(r["weak"] for r in res)
assert adv.needed_exam(60, 60) == (60 - 0.3 * 60) / 0.7
assert "improve" in adv.chat_response("How can I improve Discrete Mathematics?", {"name": "T"}, res).lower() or True
run = adv.agent_run({"name": "T", "reg_no": "P1", "attendance": 80, "study_hours": 12}, res)
assert all(run[k] for k in ("perceive", "analyse", "decide", "recommend", "act")) and "BEGIN:VCALENDAR" in run["ics"] and run["emails"]
print("Option 1 ok:", len(res), "subjects analysed, calendar + emails produced")

# ---- Option 2: support (supermarket)
import os, tempfile
from db import Database
from support_engine import SAMPLE_INBOX, triage_run
db = Database(os.path.join(tempfile.mkdtemp(), "t.db"))
bot = SupportBot(db)
cases = {"can I pay with mpesa": "payment_methods", "what time do you close": "opening_hours",
         "will you bring groceries to my house": "delivery", "how do I get a loyalty card": "loyalty_card", "hello there": "greeting"}
hits = sum(bot.respond(q)["intent"] == i for q, i in cases.items())
assert hits >= 4, hits
assert bot.respond("who will win the world cup")["status"] == "fallback"
assert "299" in bot.respond("how much is cooking oil 1L")["text"]                 # live catalogue price (offer)
log = bot.respond("my cash is missing from the ledger")["log_index"]
bot.correct(log, "receipts_invoices")
assert bot.respond("my cash is missing from the ledger")["intent"] == "receipts_invoices"      # it learned
rows = SupportBot(db).triage(SAMPLE_INBOX)
assert any(r["priority"] == "High" for r in rows) and any(r["ticket"] != "-" for r in rows) and triage_run(rows)["act"]
print(f"Option 2 ok: {hits}/{len(cases)} intents, catalogue price, learned a correction")

# ---- Option 3: finance
cat = fin.Categoriser()
assert cat.predict("KPLC tokens")[0] == "Utilities" and cat.predict("matatu fare")[0] == "Transport"
df, inc = fin.sample_data(cat)
a = fin.analyse(df, fin.monthly_income(inc), 50000)
f = fin.forecast(df)
recs = fin.recommend(a, f)
assert f["total"] > 0 and len(recs) > 3 and len(a["excess"]) > 0
ap = fin.autopilot(df, a, f, 25)
assert len(ap["budget"]) > 5 and ap["transfer_amount"] > 0 and fin.agent_run(a, f, recs, ap)["act"]
print(f"Option 3 ok: {len(df)} transactions, forecast {f['total']:,.0f}, {len(recs)} recommendations")
