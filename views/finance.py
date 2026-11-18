"""Option 3 - AI-Powered Personal Finance Management Agent (page)."""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import finance_engine as fe
import services
import theme

P = theme.pal()
PRIORITY_TONE = {"High": "bad", "Medium": "warn", "Low": "info"}

theme.hero("Finance Agent",
           "Record income and daily spending. The agent categorises every expense, finds patterns and waste, "
           "predicts next month and recommends how to save.")
services.source_button("finance")

ss = st.session_state
if "fin_cat" not in ss:
    ss.fin_cat = fe.Categoriser(services.database(), services.gemini())
    ss.fin_tx = fe.empty_tx()
    ss.fin_income = []
cat: fe.Categoriser = ss.fin_cat
cat.gemini = services.gemini()        # picks up a key pasted after the page loaded


def load_sample():
    ss.fin_tx, ss.fin_income = fe.sample_data(cat)


def clear_all():
    ss.fin_tx, ss.fin_income = fe.empty_tx(), []


def recategorise():
    """Re-run the classifier on every transaction the user has not corrected by hand."""
    df = ss.fin_tx
    mask = ~df["manual"]
    if mask.any():
        res = cat.predict_many(df.loc[mask, "description"])
        df.loc[mask, "category"] = [r[0] for r in res]
        df.loc[mask, "confidence"] = [r[1] for r in res]


t_in, t_over, t_pat, t_fc, t_rec, t_auto, t_goal, t_rep = st.tabs([
    ":material/payments: Income & expenses", ":material/pie_chart: Overview", ":material/search_insights: Patterns",
    ":material/trending_up: Forecast", ":material/savings: Recommendations", ":material/smart_toy: Autopilot",
    ":material/flag: Savings goal", ":material/description: Report"])

# --------------------------------------------------------- income & expenses
with t_in:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Income")
        cur = st.text_input("Currency", "KES", key="fin_cur", max_chars=5)
        with st.form("fin_income_form", clear_on_submit=True):
            src = st.text_input("Source", placeholder="Salary, business, allowance")
            c1, c2 = st.columns(2)
            amt = c1.number_input("Amount", min_value=0.0, value=None, step=500.0)
            freq = c2.selectbox("How often", ["Monthly", "Weekly", "Yearly"])
            ok = st.form_submit_button("Add income", icon=":material/add:", type="primary")
        if ok:
            if src.strip() and amt:
                ss.fin_income.append({"source": src.strip(), "amount": float(amt), "frequency": freq})
                st.rerun()
            else:
                theme.callout("Enter a source and an amount.", "warn")
        if ss.fin_income:
            theme.html_table(pd.DataFrame([{"Source": s["source"], "Amount": f"{s['amount']:,.0f}", "Frequency": s["frequency"]}
                                           for s in ss.fin_income]), num_cols=("Amount",))
            rm = st.selectbox("Remove an income source", [s["source"] for s in ss.fin_income], index=None,
                              placeholder="Remove an income source", label_visibility="collapsed", key="fin_inc_rm")
            if rm:
                ss.fin_income = [s for s in ss.fin_income if s["source"] != rm]
                st.rerun()
        savings = st.number_input("Savings you already have", min_value=0.0, value=0.0, step=1000.0, key="fin_cursav",
                                  help="Used for the emergency-fund and goal advice. It stays in this session only.")
        st.button("Load sample data", icon=":material/dataset:", on_click=load_sample, width="stretch")
        st.button("Clear everything", icon=":material/delete_sweep:", on_click=clear_all, width="stretch")

    with right:
        st.subheader("Daily expenditure")
        with st.form("fin_exp_form", clear_on_submit=True):
            e1, e2, e3 = st.columns([1.2, 2, 1])
            d = e1.date_input("Date", dt.date.today(), key="fin_date")
            desc = e2.text_input("What did you spend on?", placeholder="e.g. Naivas, matatu fare, KPLC tokens")
            val = e3.number_input("Amount", min_value=0.0, value=None, step=50.0)
            add = st.form_submit_button("Add expense", icon=":material/add:", type="primary")
        if add:
            if desc.strip() and val:
                ss.fin_tx = fe.add_transactions(ss.fin_tx, [(d, desc, val)], cat)
                row = ss.fin_tx[ss.fin_tx["description"] == desc.strip()].iloc[-1]
                theme.callout(f"Categorised automatically as {row['category']} ({row['confidence']:.0%} sure).", "good")
            else:
                theme.callout("Enter a description and an amount.", "warn")
        with st.expander("Import many expenses at once", icon=":material/upload_file:"):
            st.caption("One per line: `date, description, amount` (date optional). Example: `2026-11-03, Naivas, 1450`")
            txt = st.text_area("Paste lines", height=110, key="fin_paste", label_visibility="collapsed")
            up = st.file_uploader("or upload a CSV / text file", type=["csv", "txt"], key="fin_upload")
            if st.button("Import", icon=":material/download:", key="fin_import"):
                body = txt + "\n" + (up.getvalue().decode("utf-8", "ignore") if up else "")
                rows, errs = fe.parse_text(body)
                if rows:
                    ss.fin_tx = fe.add_transactions(ss.fin_tx, rows, cat)
                    theme.callout(f"Imported and categorised {len(rows)} expense(s).", "good")
                for e in errs[:5]:
                    theme.callout(e, "warn")

        df = ss.fin_tx
        if df.empty:
            theme.callout("No expenses yet. Add one above or load the sample data.", "info")
        else:
            st.markdown(f"**Latest transactions** ({len(df)} recorded)")
            show = df.tail(15).iloc[::-1].copy()
            show["Check"] = ((show["confidence"] < 0.5) & ~show["manual"]).map({True: "Please check", False: ""})
            tbl = pd.DataFrame({"Date": show["date"].dt.date.astype(str), "Description": show["description"],
                                "Amount": show["amount"].map(lambda v: f"{v:,.0f}"), "Category": show["category"],
                                "Sure": show["confidence"].map(lambda v: f"{v:.0%}"), "Check": show["Check"]})
            theme.html_table(tbl, num_cols=("Amount", "Sure"), tone_col="Check", tones={"Please check": "warn"})

            st.markdown("**Teach the agent**")
            st.caption("If a category is wrong, fix it. The agent retrains and re-categorises similar entries.")
            recent = list(df.tail(40).index[::-1])
            c1, c2, c3 = st.columns([3, 2, 1])
            pick = c1.selectbox("Transaction", recent, key="fin_fix_tx", label_visibility="collapsed",
                                format_func=lambda i: f"{df.at[i, 'date'].date()}  {df.at[i, 'description']}  ({df.at[i, 'amount']:,.0f})")
            newcat = c2.selectbox("Correct category", fe.CATEGORIES, key="fin_fix_cat", label_visibility="collapsed")
            if c3.button("Correct", icon=":material/school:", key="fin_fix_btn", width="stretch"):
                cat.learn(df.at[pick, "description"], newcat)
                ss.fin_tx.at[pick, "category"], ss.fin_tx.at[pick, "confidence"], ss.fin_tx.at[pick, "manual"] = newcat, 1.0, True
                recategorise()
                st.rerun()

# ------------------------------------------------------------------ analysis
df = ss.fin_tx
inc_m = fe.monthly_income(ss.fin_income)
NO_DATA = "Add some expenses in the Income & expenses tab, or load the sample data."
a = f = recs = None
if not df.empty:
    a = fe.analyse(df, inc_m, savings)
    f = fe.forecast(df)
    recs = fe.recommend(a, f)
money = lambda v: f"{cur} {v:,.0f}"


def need_data(tab):
    with tab:
        theme.callout(NO_DATA, "info")


def short_notice():
    if a["days"] < 14:
        theme.callout(f"Only {a['days']} days of data so far. Estimates become reliable after about two weeks.", "warn")
    if not inc_m:
        theme.callout("Add your income to see savings rate, budgets and goal advice.", "warn")


# ------------------------------------------------------------------ overview
if a is None:
    for t in (t_over, t_pat, t_fc, t_rec, t_auto, t_goal, t_rep):
        need_data(t)
else:
    with t_over:
        short_notice()
        top = a["by_cat"].index[0]
        rate = a["savings_rate"]
        theme.cards([("Monthly income", money(inc_m) if inc_m else "-", "from your sources"),
                     ("Monthly spending", money(a["spend_m"]), "estimated from your records"),
                     ("Savings rate", f"{rate:.0%}" if inc_m else "-", "target 20% or more",
                      None if not inc_m else "good" if rate >= 0.2 else "warn" if rate >= 0.05 else "bad"),
                     ("Biggest category", top, f"{a['by_cat'].iloc[0]['share']:.0%} of spending")])
        st.write("")
        c1, c2 = st.columns([3, 2])
        with c1:
            bc = a["by_cat"].iloc[::-1]
            fig = go.Figure(go.Bar(y=bc.index, x=bc["monthly"], orientation="h", marker_color=P["series"][0],
                                   marker_cornerradius=4, text=[f"{v:,.0f}" for v in bc["monthly"]], textposition="outside",
                                   textfont=dict(color=P["text"])))
            fig.update_layout(title=f"Spending by category ({cur} per month)", xaxis_range=[0, bc["monthly"].max() * 1.2])
            theme.show(theme.style_fig(fig, 80 + 34 * len(bc), legend=False), "fin_cat")
        with c2:
            groups = ["Needs", "Wants", "Savings"]
            fig = go.Figure()
            fig.add_bar(y=groups[::-1], x=[a["pct"].get(g, 0) for g in groups][::-1], orientation="h", name="You",
                        marker_color=P["series"][0], marker_cornerradius=4)
            fig.add_bar(y=groups[::-1], x=[50, 30, 20][::-1], orientation="h", name="50/30/20 guide",
                        marker_color=P["series"][1], marker_cornerradius=4)
            fig.update_layout(title="Share of income", barmode="group", bargap=0.35, xaxis_title="% of income")
            fig.update_traces(width=0.3)
            theme.show(theme.style_fig(fig, 300), "fin_503020")
            st.caption("Needs: housing, food, transport, utilities, health, education. Wants: eating out, shopping, "
                       "entertainment, subscriptions.")
        mo = a["monthly"]
        if len(mo) >= 2:
            labels = [str(p) for p in mo.index]
            fig = go.Figure(go.Bar(x=labels, y=mo.values, marker_color=P["series"][0], marker_cornerradius=4, name="Spending"))
            if inc_m:
                fig.add_hline(y=inc_m, line_dash="dot", line_color=P["good"], annotation_text="Income",
                              annotation_font_color=P["muted"])
            fig.update_layout(title="Spending per month (first and last month may be partial)")
            theme.show(theme.style_fig(fig, 300, legend=False), "fin_month")
        if inc_m:
            if rate < 0:
                theme.callout("You are spending more than you earn. See the Recommendations tab.", "bad")
            elif rate < 0.2:
                theme.callout(f"You keep {rate:.0%} of your income. The agent suggests aiming for at least 20%.", "warn")
            else:
                theme.callout(f"You keep {rate:.0%} of your income, which meets the 20% guideline.", "good")

    # ---------------------------------------------------------------- patterns
    with t_pat:
        pat = a["pat"]
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            dow = pat["dow"]
            peak = int(dow.idxmax())
            fig = go.Figure(go.Bar(x=fe.DAYS, y=dow.values, marker_color=[P["series"][1] if i == peak else P["series"][0] for i in range(7)],
                                   marker_cornerradius=4, text=[f"{v:,.0f}" for v in dow.values], textposition="outside",
                                   textfont=dict(color=P["text"])))
            fig.update_layout(title="Average daily spending by weekday (excluding rent, school and subscriptions)",
                              yaxis_range=[0, dow.max() * 1.2])
            theme.show(theme.style_fig(fig, 340, legend=False), "fin_dow")
        with c2:
            st.markdown("**What the agent noticed**")
            theme.callout(f"Your highest-spending day is {pat['peak_day']}. Weekends run at {pat['weekend_ratio']:.1f}x weekdays.",
                          "warn" if pat["weekend_ratio"] >= 1.4 else "info")
            tr = pat["trend"]
            theme.callout(f"Daily spending is {'up' if tr > 0 else 'down'} {abs(tr) * 100:.0f}% in the second half of your records.",
                          "bad" if tr > 0.1 else "good" if tr < -0.05 else "info")
            lk = pat["leak"]
            if lk["n"] >= 20:
                theme.callout(f"{lk['n']} small purchases (each {lk['threshold']:,.0f} or less) add up to {lk['share']:.0%} of "
                              f"spending, about {money(lk['monthly'])} a month.", "warn" if lk["share"] >= 0.10 else "info")
        st.markdown("**Recurring payments**")
        rec = pat["recurring"]
        if len(rec):
            t = rec.copy()
            t["Typical amount"] = t["Typical amount"].map(lambda v: f"{v:,.0f}")
            t["Per month"] = t["Per month"].map(lambda v: f"{v:,.0f}")
            theme.html_table(t, num_cols=("Typical amount", "Per month"))
        else:
            st.caption("No recurring payments detected yet.")
        st.markdown("**Excessive or unnecessary spending**")
        ex = a["excess"]
        if len(ex):
            t = pd.DataFrame({"Category": ex["Category"], "Type": ex["Type"], "Per month": ex["Monthly"].map(lambda v: f"{v:,.0f}"),
                              "Guideline": ex["Guideline"].map(lambda v: f"{v:,.0f}"), "Over by": ex["Over by"].map(lambda v: f"{v:,.0f}"),
                              "Verdict": ex["Type"].map({"Wants": "Discretionary, cut back", "Needs": "Essential, trim"})})
            theme.html_table(t, num_cols=("Per month", "Guideline", "Over by"), tone_col="Verdict",
                             tones={"Discretionary, cut back": "bad", "Essential, trim": "warn"})
            st.caption("Guidelines are standard percentages of income per category. Adjust them to your own situation.")
        else:
            theme.callout("No category is above its guideline." if inc_m else "Add income to compare with guidelines.", "good" if inc_m else "info")
        st.markdown("**Unusual transactions**")
        un = pat["unusual"]
        if len(un):
            t = un.copy(); t["Amount"] = t["Amount"].map(lambda v: f"{v:,.0f}")
            theme.html_table(t, num_cols=("Amount",))
        else:
            st.caption("Nothing unusual found.")

    # ---------------------------------------------------------------- forecast
    with t_fc:
        short_notice()
        pred_save = inc_m - f["total"] * (fe.MONTH / f["horizon"]) if inc_m else None
        theme.cards([("Predicted spending", money(f["total"]), f"next {f['horizon']} days"),
                     ("Likely range", f"{money(f['low'])} to {money(f['high'])}", "80% interval"),
                     ("Predicted savings", "-" if pred_save is None else money(pred_save), "income minus predicted spending",
                      None if pred_save is None else "good" if pred_save > 0 else "bad"),
                     ("Model used", f["model"], "chosen by back-testing")])
        st.write("")
        hist = f["hist"].iloc[-60:]
        fig = go.Figure()
        fig.add_scatter(x=hist.index, y=hist.values, mode="lines", name="Daily spending (7-day average)",
                        line=dict(color=P["series"][0], width=2))
        fig.add_scatter(x=f["dates"], y=f["smooth_high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=f["dates"], y=f["smooth_low"], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(235,104,52,0.20)", name="Likely range", hoverinfo="skip")
        fig.add_scatter(x=f["dates"], y=f["smooth_fc"], mode="lines", name="Forecast",
                        line=dict(color=P["series"][1], width=2, dash="dash"))
        fig.update_layout(title=f"Everyday spending: history and forecast ({cur} per day, excludes rent, school fees, subscriptions)")
        theme.show(theme.style_fig(fig, 380), "fin_fc")
        if f["mae_model"] is not None:
            st.caption(f"Back-test on the last 7 days: regression error {f['mae_model']:,.0f} per day versus "
                       f"{f['mae_base']:,.0f} for a simple moving average. The better method is used.")
        st.markdown("**Predicted spending by category**")
        rows = []
        for c, v in sorted(f["per_cat"].items(), key=lambda x: -x[1]):
            typical = a["by_cat"]["monthly"].get(c, 0) * f["horizon"] / fe.MONTH
            ch = (v - typical) / typical if typical else 0
            rows.append({"Category": c, "Typical for the period": f"{typical:,.0f}", "Predicted": f"{v:,.0f}",
                         "Change": f"{ch:+.0%}"})
        theme.html_table(pd.DataFrame(rows), num_cols=("Typical for the period", "Predicted", "Change"))

    # ---------------------------------------------------------- recommendations
    with t_rec:
        cuts = sum(r["saving"] for r in recs if r["title"].startswith("Bring"))
        gap_to_20 = max(0.2 * inc_m - max(a["surplus_m"], 0), 0) if inc_m else 0
        theme.cards([("Savings opportunity", money(cuts), "per month from category cuts", "good"),
                     ("Gap to a 20% savings rate", money(gap_to_20) if inc_m else "-", "per month"),
                     ("Recommendations", str(len(recs)), "ranked by impact")])
        st.write("")
        for r in recs:
            with st.container(border=True):
                head = st.columns([5, 2])
                head[0].markdown(f"**{r['title']}**")
                head[1].markdown(theme.badge(f"{r['priority']} priority", PRIORITY_TONE[r["priority"]]), unsafe_allow_html=True)
                st.markdown(r["detail"])
                if r["saving"]:
                    st.caption(f"Potential saving: up to {money(r['saving'])} per month")
        st.write("")
        if st.button("Ask Gemini for a plain-language plan", icon=":material/psychology:", key="fin_gem_plan",
                     disabled=not services.gemini().available(),
                     help="Needs a Gemini API key (sidebar). Gemini only rewrites the agent's own numbers."):
            with st.spinner("Asking Gemini..."):
                ss.fin_gem = fe.gemini_plan(services.gemini(), a, f, recs)
            if not ss.fin_gem:
                theme.callout("Gemini did not answer: " + services.gemini().last_error, "warn")
        if ss.get("fin_gem"):
            theme.callout(ss.fin_gem["text"], "info")
        elif not services.gemini().available():
            st.caption("Add a Gemini key in the sidebar to get an AI-written plan. The agent's own recommendations above work without it.")

    # --------------------------------------------------------------- autopilot
    with t_auto:
        short_notice()
        st.subheader("What the agent does on its own")
        st.caption("Without being asked, the agent sets budgets, watches the month, warns you before you overspend, "
                   "schedules your saving and prepares disputes for double charges.")
        payday = st.number_input("Your payday (day of the month)", 1, 28, 25, key="fin_payday")
        ap = fe.autopilot(df, a, f, int(payday))
        theme.agent_loop(fe.agent_run(a, f, recs, ap, cur))
        st.subheader("Budgets the agent set")
        bt = ap["budget"].copy()
        bt = pd.DataFrame({"Category": bt["Category"], "Typical month": bt["Typical"].map(lambda v: f"{v:,.0f}"),
                           "New budget": bt["Budget"].map(lambda v: f"{v:,.0f}"),
                           "Change": bt["Change"].map(lambda v: f"{v:+,.0f}"), "Reason": bt["Why"]})
        theme.html_table(bt, num_cols=("Typical month", "New budget", "Change"))
        st.subheader(f"Month-to-date watch ({ap['month']}, day {ap['elapsed']})")
        if ap["alerts"]:
            for al in ap["alerts"]:
                theme.callout(al["Message"], "bad" if al["Spent so far"] >= al["Budget"] else "warn")
        elif ap["elapsed"] < 7:
            st.caption("The agent starts watching after a week of the month has passed.")
        else:
            theme.callout("Every everyday category is on pace to stay within its budget.", "good")
        st.subheader("Saving transfers scheduled")
        theme.cards([("Monthly transfer", money(ap["transfer_amount"]), "pay yourself first", "good"),
                     ("Next transfers", ", ".join(f"{d:%d %b}" for d in ap["transfer_dates"]), f"on day {ap['payday']}"),
                     ("Needs", "following the new budgets", "the transfer assumes the trimmed categories are respected")])
        dups = a["spend"][a["spend"].duplicated(["date", "description", "amount"], keep="first")]
        if len(dups):
            st.subheader("Duplicate charges flagged")
            for i, row in dups.iterrows():
                with st.expander(f"{row['date']:%d %b}: {row['description']} ({row['amount']:,.0f})"):
                    st.text_area("Draft message", fe.draft_dispute(st.session_state.get("fin_owner", ""), cur, row),
                                 height=190, key=f"fin_disp_{i}", label_visibility="collapsed")

    # -------------------------------------------------------------------- goal
    with t_goal:
        st.subheader("Savings goal planner")
        g1, g2, g3, g4 = st.columns(4)
        gname = g1.text_input("Goal", "Emergency fund", key="fin_gname")
        gamt = g2.number_input("Target amount", min_value=1000.0, value=100000.0, step=5000.0, key="fin_gamt")
        gmon = g3.number_input("Months to reach it", min_value=1, value=12, step=1, key="fin_gmon")
        gsaved = g4.number_input("Already saved", min_value=0.0, value=float(savings), step=1000.0, key="fin_gsaved")
        plan = fe.goal_plan(gamt, int(gmon), gsaved, a["surplus_m"], a["excess"])
        theme.cards([("Still to save", money(plan["need"]), gname), ("Needed per month", money(plan["per_month"]), f"over {int(gmon)} months"),
                     ("Your monthly surplus", money(a["surplus_m"]), "income minus spending", "good" if a["surplus_m"] > 0 else "bad")])
        st.write("")
        if not inc_m:
            theme.callout("Add your income to check whether this goal is affordable.", "warn")
        elif plan["feasible"]:
            theme.callout(f"Achievable. Your current surplus already covers {money(plan['per_month'])} a month.", "good")
        else:
            theme.callout(f"You are short by about {money(plan['gap'])} a month.", "warn")
            if plan["cuts"]:
                st.markdown("**Where the agent would find it**")
                theme.html_table(pd.DataFrame([{"Cut back on": c, "Monthly saving": f"{v:,.0f}"} for c, v in plan["cuts"]]),
                                 num_cols=("Monthly saving",))
            if plan["remaining_gap"] > 0:
                longer = plan["need"] / max(a["surplus_m"] + sum(v for _, v in plan["cuts"]), 1)
                theme.callout(f"Even after those cuts you would be {money(plan['remaining_gap'])} a month short. "
                              f"Extending the goal to about {longer:.0f} months, or raising income, would make it realistic.", "info")

    # ------------------------------------------------------------------ report
    with t_rep:
        owner = st.text_input("Name on the report", key="fin_owner")
        html = fe.report_html(owner, cur, a, f, recs)
        c1, c2 = st.columns(2)
        c1.download_button("Download summary (HTML, print to PDF)", html, file_name="financial_summary.html", mime="text/html",
                           icon=":material/download:", type="primary", width="stretch")
        c2.download_button("Download categorised transactions (CSV)", df.to_csv(index=False), file_name="transactions_categorised.csv",
                           mime="text/csv", icon=":material/table_view:", width="stretch")
        try:
            st.iframe(html, height=720)
        except Exception:
            import streamlit.components.v1 as components
            components.html(html, height=720, scrolling=True)
