"""Overview: the three AI agents, what each one perceives / decides / does, a one-click demo and honest metrics."""
import pandas as pd
import streamlit as st

import advisor_engine as adv
import finance_engine as fin
import services
import theme
from support_engine import SAMPLE_INBOX, SupportBot, triage_run

ss = st.session_state

theme.hero("AI Agents Suite",
           "Three intelligent agents that perceive information, analyse situations, make decisions, solve problems, "
           "recommend actions and carry out tasks on their own.")

# ------------------------------------------------------------------ status
db = services.database()
g = services.gemini()
theme.cards([("Group members", f"{len(db.list_members())} of 6", "stored in the database"),
             ("Research mode", "Gemini" if g.available() else "Wikipedia", "Google Search grounding" if g.available() else "add a Gemini key in the sidebar"),
             ("Remembered research", str(db.cache_count()), "units and questions cached"),
             ("Tests", (lambda t: f"{sum(r['passed'] for r in t)} of {len(t)}" if t else "not run")(db.load_tests()), "see Testing and data")])
st.write("")

st.subheader("The three agents")
c1, c2, c3 = st.columns(3)
for col, title, line, ai, ag in [
    (c1, "Academic Advisor", "Predicts final marks, finds weak units and researches learning resources for any unit name.",
     "Ridge regression, expert-system rules, research", "advisor"),
    (c2, "Supermarket Support Agent", "Answers customers from its knowledge base and live product catalogue, and researches what it does not know.",
     "TF-IDF + Logistic Regression, retrieval, research, feedback learning", "support"),
    (c3, "Finance Agent", "Categorises spending, spots waste, forecasts the month and runs an autopilot of budgets and alerts.",
     "Text classification, pattern detection, forecasting", "finance")]:
    col.markdown(f'<div class="ui-card"><div class="l">Agent</div><div class="v" style="font-size:1.1rem">{title}</div>'
                 f'<p style="margin:8px 0;color:var(--text)">{line}</p><div class="s"><b>AI used:</b> {ai}</div></div>',
                 unsafe_allow_html=True)
    with col:
        services.source_button(ag)
st.caption("Design, techniques, setup and testing are explained in README.md and in the PDF report on the Group and report page.")

# ------------------------------------------------------------------ one-click demo
st.subheader("Autonomous demo")
st.caption("Runs all three agents on built-in sample data, with no clicks or typing, and shows what each one concluded and did.")


def run_demo() -> dict:
    prof = {"name": "Sample Student", "reg_no": "P101/0001/2024", "course": "BSc Computer Science",
            "attendance": 80.0, "study_hours": 12.0}
    res = adv.analyse(adv.sample_marks(), 80, 12)
    adv_run = adv.agent_run(prof, res)

    bot = SupportBot(services.database(), services.researcher())
    sup_run = triage_run(bot.triage(SAMPLE_INBOX))

    cat = fin.Categoriser(services.database(), services.gemini())
    df, inc = fin.sample_data(cat)
    a = fin.analyse(df, fin.monthly_income(inc), 50000)
    f = fin.forecast(df)
    recs = fin.recommend(a, f)
    ap = fin.autopilot(df, a, f, 25)
    fin_run = fin.agent_run(a, f, recs, ap)

    mr = adv.model_report()
    sh = bot.evaluate_heldout()
    cv = bot.evaluate()
    fh = fin.categoriser_heldout(cat)
    metrics = [
        {"Agent": "Academic Advisor", "Model": "Ridge regression (final mark)", "Measure": "R-squared / mean error on 1,000 unseen records",
         "Result": f"{mr['r2']:.2f} / {mr['mae']:.1f} marks", "Caveat": "Synthetic data, replace with real marks"},
        {"Agent": "Support Agent", "Model": "TF-IDF + Logistic Regression", "Measure": f"Accuracy on {sh['n']} fresh phrasings",
         "Result": f"{sh['accuracy']:.0%}", "Caveat": "Small test set, small knowledge base"},
        {"Agent": "Support Agent", "Model": "Same model", "Measure": "5-fold cross-validation on training phrases",
         "Result": f"{cv['accuracy']:.0%}" if cv["accuracy"] else "-", "Caveat": "Uses the training phrasings"},
        {"Agent": "Finance Agent", "Model": "TF-IDF + Logistic Regression (categories)", "Measure": f"Accuracy on {fh['n']} fresh descriptions",
         "Result": f"{fh['accuracy']:.0%}", "Caveat": "Learns from corrections"},
        {"Agent": "Finance Agent", "Model": f["model"], "Measure": "Back-test: error per day on the last 7 days",
         "Result": (f"{f['mae_model']:,.0f} vs {f['mae_base']:,.0f} baseline" if f["mae_model"] is not None else "-"),
         "Caveat": "Sample data is generated"},
    ]
    return {"adv": adv_run, "sup": sup_run, "fin": fin_run, "metrics": metrics, "support_wrong": sh["wrong"],
            "fin_wrong": fh["wrong"]}


if st.button("Run all three agents", icon=":material/play_arrow:", type="primary", key="home_demo_btn"):
    with st.spinner("The agents are working..."):
        ss.home_demo = run_demo()

demo = ss.get("home_demo")
if demo:
    t1, t2, t3, t4 = st.tabs([":material/school: Academic Advisor", ":material/support_agent: Support Agent",
                              ":material/account_balance_wallet: Finance Agent", ":material/fact_check: How well it works"])
    with t1:
        theme.agent_loop(demo["adv"])
    with t2:
        theme.agent_loop(demo["sup"])
    with t3:
        theme.agent_loop(demo["fin"])
    with t4:
        theme.html_table(pd.DataFrame(demo["metrics"]))
        theme.callout("These numbers are measured, not promised. The test phrasings were not used for training and the agents were "
                      "not tuned to them. The advisor and finance data are synthetic, so real data would be needed for real claims.",
                      "info")
        if demo["support_wrong"]:
            st.markdown("**Where the support agent was wrong**")
            theme.html_table(pd.DataFrame([{"Phrase": t, "Expected": e, "Predicted": g} for t, e, g in demo["support_wrong"]]))
        if demo["fin_wrong"]:
            st.markdown("**Where the finance categoriser was wrong**")
            theme.html_table(pd.DataFrame([{"Description": t, "Expected": e, "Predicted": g} for t, e, g in demo["fin_wrong"]]))
