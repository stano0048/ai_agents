"""Command-line interface for the three agents (the handout accepts a GUI, web or command-line interface).

  python cli.py advisor --sample                 analyse the sample marks
  python cli.py advisor --unit "Entrepreneurship" research a unit by name
  python cli.py support "do you deliver?"        one question (omit the text for an interactive chat)
  python cli.py finance --sample                 analyse the sample transactions
  python cli.py members                          list group members
  python cli.py report group_report.pdf          write the PDF report
Set GEMINI_API_KEY to let the agents use Gemini; DB_PATH to choose the database file.
"""
import argparse
import sys

from db import Database
from llm import Gemini
from research import Researcher


def main(argv=None):
    ap = argparse.ArgumentParser(description="AI Agents Suite command line")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("advisor"); a.add_argument("--sample", action="store_true"); a.add_argument("--unit"); a.add_argument("--ask")
    s = sub.add_parser("support"); s.add_argument("text", nargs="*")
    f = sub.add_parser("finance"); f.add_argument("--sample", action="store_true")
    sub.add_parser("members")
    r = sub.add_parser("report"); r.add_argument("out", nargs="?", default="group_report.pdf")
    args = ap.parse_args(argv)

    db = Database()
    res = Researcher(db, Gemini(db=db))
    print(f"[research mode: {'Gemini' if res.gemini.available() else 'Wikipedia / offline'}]")

    if args.cmd == "advisor":
        import advisor_engine as eng
        if args.unit:
            x = res.research_unit(args.unit)
            print(f"{x['unit']} ({x['mode']})\n{x['summary']}")
            for i in x["resources"]:
                print(f" - {i['title']}: {i['url']}")
            return 0
        results = eng.analyse(eng.sample_marks(), 80, 12)
        prof = {"name": "Sample Student", "course": "BSc Computer Science", "attendance": 80, "study_hours": 12}
        if args.ask:
            print(eng.chat_response(args.ask, prof, results, res)); return 0
        for r_ in results:
            print(f"{r_['subject']:<32} predicted {r_['predicted']:>3.0f}%  grade {r_['grade']}  risk {r_['risk']}")
        run = eng.agent_run(prof, results)
        for stage in ("decide", "recommend", "act"):
            print(f"\n{stage.upper()}"); [print(" -", x) for x in run[stage]]
    elif args.cmd == "support":
        from support_engine import SupportBot
        bot = SupportBot(db, res)
        if args.text:
            print(bot.respond(" ".join(args.text))["text"]); return 0
        print("Supermarket support agent. Type 'quit' to leave.")
        hist = []
        while True:
            try:
                q = input("you> ").strip()
            except EOFError:
                break
            if q.lower() in ("quit", "exit"):
                break
            r = bot.respond(q, hist)
            hist += [{"role": "user", "text": q}, {"role": "assistant", "text": r["text"]}]
            print("agent>", r["text"], f"\n[{r['status']}, confidence {r['confidence']:.2f}]")
    elif args.cmd == "finance":
        import finance_engine as fe
        cat = fe.Categoriser(db, res.gemini)
        df, inc = fe.sample_data(cat)
        an = fe.analyse(df, fe.monthly_income(inc), 50000)
        fc = fe.forecast(df)
        print(f"Spending {an['spend_m']:,.0f} a month, savings rate {an['savings_rate']:.0%}, next 30 days forecast {fc['total']:,.0f}")
        for x in fe.recommend(an, fc):
            print(f" - [{x['priority']}] {x['title']}")
    elif args.cmd == "members":
        for m in db.list_members():
            print(f"{m['name']} ({m['reg_no'] or 'no reg no'}) - {m['role']}: {m['responsibilities']}")
        if not db.list_members():
            print("No members yet. Add them on the Group page of the web app.")
    elif args.cmd == "report":
        import report
        open(args.out, "wb").write(report.build_pdf(db))
        print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
