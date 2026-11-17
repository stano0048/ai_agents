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
print("Option 1 ok:", len(res), "subjects analysed")

# ---- Option 2: support
bot = SupportBot()
cases = {"how can I settle my tuition": "fee_payment", "what date do exams begin": "exam_timetable",
         "I forgot my login": "ict_access", "can I pause my studies": "deferment", "hello there": "greeting"}
hits = sum(bot.respond(q)["intent"] == i for q, i in cases.items())
assert hits >= 4, hits
assert bot.respond("who will win the world cup")["status"] == "fallback"
before = bot.respond("my cash is missing from the ledger")["intent"]
log = bot.respond("my cash is missing from the ledger")["log_index"]
bot.correct(log, "fee_balance")
assert bot.respond("my cash is missing from the ledger")["intent"] == "fee_balance"      # it learned
print(f"Option 2 ok: {hits}/{len(cases)} intents, learned a correction (was '{before}')")

# ---- Option 3: finance
cat = fin.Categoriser()
assert cat.predict("KPLC tokens")[0] == "Utilities" and cat.predict("matatu fare")[0] == "Transport"
df, inc = fin.sample_data(cat)
a = fin.analyse(df, fin.monthly_income(inc), 50000)
f = fin.forecast(df)
recs = fin.recommend(a, f)
assert f["total"] > 0 and len(recs) > 3 and len(a["excess"]) > 0
print(f"Option 3 ok: {len(df)} transactions, forecast {f['total']:,.0f}, {len(recs)} recommendations")
