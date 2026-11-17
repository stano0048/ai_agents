"""Option 1 - AI-Powered Student Academic Advisory Agent (page)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import advisor_engine as eng
import theme

RISK_TONE = {"High": "bad", "Medium": "warn", "Watch": "info", "Low": "good"}
P = theme.pal()

theme.hero("Academic Advisor",
           "Enter your marks. The agent analyses them, predicts your final results, finds weak subjects "
           "and coaches you with a personalised plan.")

# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("adv_marks", [])
ss.setdefault("adv_chat", [])


def marks_df() -> pd.DataFrame:
    cols = ["Subject", "CAT1", "CAT2", "Assignment", "Exam"]
    return pd.DataFrame(ss.adv_marks, columns=cols) if ss.adv_marks else pd.DataFrame(columns=cols)


def load_sample():
    ss.adv_marks = eng.sample_marks().where(pd.notna(eng.sample_marks()), None).to_dict("records")
    ss.adv_chat = []


def clear_marks():
    ss.adv_marks = []
    ss.adv_chat = []


# ------------------------------------------------------------------- tabs
t_in, t_dash, t_pred, t_rec, t_chat, t_rep = st.tabs([
    ":material/person: Student & marks", ":material/monitoring: Dashboard", ":material/query_stats: Predictions",
    ":material/lightbulb: Recommendations", ":material/forum: Ask the advisor", ":material/description: Report"])

# --------------------------------------------------------- student & marks
with t_in:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Student details")
        name = st.text_input("Full name", key="adv_name")
        reg_no = st.text_input("Registration number", key="adv_reg")
        course = st.text_input("Course / programme", key="adv_course")
        c1, c2 = st.columns(2)
        attendance = c1.number_input("Attendance (%)", 0, 100, 80, key="adv_att")
        study_hours = c2.number_input("Study hours / week", 0, 80, 12, key="adv_study")
        st.caption("Attendance and study hours feed the prediction model and the rule base.")

    with right:
        st.subheader("Academic marks")
        st.caption("Marks are percentages (0-100). Coursework (CAT 1, CAT 2, assignment) counts 30%, the exam 70%. "
                   "Leave the exam empty if you have not sat it yet - the agent will predict it.")
        with st.form("adv_add", clear_on_submit=True):
            subj = st.text_input("Subject / unit name")
            f1, f2, f3, f4 = st.columns(4)
            cat1 = f1.number_input("CAT 1", 0.0, 100.0, value=None, step=1.0)
            cat2 = f2.number_input("CAT 2", 0.0, 100.0, value=None, step=1.0)
            asg = f3.number_input("Assignment", 0.0, 100.0, value=None, step=1.0)
            exam = f4.number_input("Exam", 0.0, 100.0, value=None, step=1.0)
            added = st.form_submit_button("Add subject", icon=":material/add:", type="primary")
        if added:
            if not subj.strip():
                theme.callout("Enter a subject name first.", "warn")
            elif all(v is None for v in (cat1, cat2, asg, exam)):
                theme.callout("Enter at least one mark for the subject.", "warn")
            else:
                row = {"Subject": subj.strip(), "CAT1": cat1, "CAT2": cat2, "Assignment": asg, "Exam": exam}
                ss.adv_marks = [m for m in ss.adv_marks if m["Subject"].lower() != row["Subject"].lower()] + [row]
                ss.adv_chat = []
                st.rerun()

        b1, b2, b3 = st.columns([1, 1, 1])
        b1.button("Load sample data", icon=":material/dataset:", on_click=load_sample, width="stretch")
        b2.button("Clear all marks", icon=":material/delete_sweep:", on_click=clear_marks, width="stretch")
        if ss.adv_marks:
            to_remove = b3.selectbox("Remove subject", [m["Subject"] for m in ss.adv_marks],
                                     label_visibility="collapsed", index=None, placeholder="Remove a subject")
            if to_remove:
                ss.adv_marks = [m for m in ss.adv_marks if m["Subject"] != to_remove]
                ss.adv_chat = []
                st.rerun()

        show = marks_df().copy()
        for c in ["CAT1", "CAT2", "Assignment", "Exam"]:
            show[c] = show[c].apply(lambda v: "-" if v is None or pd.isna(v) else f"{v:.0f}")
        theme.html_table(show, num_cols=("CAT1", "CAT2", "Assignment", "Exam"))

profile = {"name": name, "reg_no": reg_no, "course": course,
           "attendance": float(attendance), "study_hours": float(study_hours)}
results = eng.analyse(marks_df(), attendance, study_hours)
summ = eng.overall_summary(results)
NO_DATA = "Add at least one subject in the Student & marks tab, or load the sample data."

# --------------------------------------------------------------- dashboard
with t_dash:
    if not results:
        theme.callout(NO_DATA, "info")
    else:
        theme.cards([
            ("Projected average", f"{summ['average']:.0f}%", "across all subjects"),
            ("Projected standing", summ["classification"].split(" (")[0], "from the average"),
            ("Weak subjects", f"{summ['n_weak']} of {summ['n_subjects']}", "medium or high risk",
             "bad" if summ["n_weak"] else "good"),
            ("Strongest subject", summ["best"]["subject"], f"{summ['best']['predicted']:.0f}%", "good"),
        ])
        st.write("")
        names = [r["subject"] for r in results]
        a, b = st.columns([3, 2])
        with a:
            fig = go.Figure()
            fig.add_bar(y=names[::-1], x=[r["ca_avg"] for r in results][::-1], name="Coursework average",
                        orientation="h", marker_color=P["series"][0], marker_cornerradius=4)
            fig.add_bar(y=names[::-1], x=[r["predicted"] for r in results][::-1], name="Final / predicted",
                        orientation="h", marker_color=P["series"][1], marker_cornerradius=4)
            fig.add_vline(x=40, line_dash="dot", line_color=P["bad"], annotation_text="Pass 40",
                          annotation_font_color=P["muted"])
            fig.add_vline(x=70, line_dash="dot", line_color=P["good"], annotation_text="A 70",
                          annotation_font_color=P["muted"])
            fig.update_layout(title="Performance by subject", barmode="group", bargap=0.35, xaxis_range=[0, 100])
            fig.update_traces(width=0.32)
            theme.show(theme.style_fig(fig, 90 + 70 * len(results)), "adv_bar")
        with b:
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=summ["average"], number={"suffix": "%", "font": {"color": P["text"]}},
                title={"text": "Overall projection", "font": {"color": P["text"], "size": 15}},
                gauge={"axis": {"range": [0, 100], "tickcolor": P["muted"], "tickfont": {"color": P["muted"]}},
                       "bar": {"color": P["accent"]}, "bgcolor": P["surface"], "bordercolor": P["border"],
                       "steps": [{"range": [0, 40], "color": P["grid"]}, {"range": [40, 100], "color": P["bg"]}],
                       "threshold": {"line": {"color": P["bad"], "width": 3}, "value": 40}}))
            theme.show(theme.style_fig(g, 380, legend=False), "adv_gauge")
        if len(results) >= 3:
            vals = [r["predicted"] for r in results]
            radar = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], fill="toself",
                                              line_color=P["series"][0], fillcolor="rgba(42,120,214,0.18)"))
            radar.update_layout(title="Strengths and weaknesses",
                                polar=dict(bgcolor="rgba(0,0,0,0)",
                                           radialaxis=dict(range=[0, 100], gridcolor=P["grid"],
                                                           tickfont=dict(color=P["muted"]), linecolor=P["grid"]),
                                           angularaxis=dict(gridcolor=P["grid"], tickfont=dict(color=P["text"]))))
            theme.show(theme.style_fig(radar, 400, legend=False), "adv_radar")
        weak = sorted([r for r in results if r["weak"]], key=lambda r: -r["priority"])
        if weak:
            theme.callout("Weak subjects detected: " + ", ".join(f"{r['subject']} ({r['predicted']:.0f}%)" for r in weak), "bad")
        else:
            theme.callout("No weak subjects detected. Keep it up.", "good")

# ------------------------------------------------------------- predictions
with t_pred:
    if not results:
        theme.callout(NO_DATA, "info")
    else:
        st.subheader("Predicted final marks")
        st.caption("A regression model uses coursework average, mark trend, attendance and study hours. "
                   "The bar shows the likely (80%) range. Subjects with an exam mark show their final result.")
        fig = go.Figure()
        for r in results:
            fig.add_trace(go.Scatter(
                x=[r["predicted"]], y=[r["subject"]], mode="markers",
                marker=dict(size=13, color=P["series"][0] if r["status"] == "Predicted" else P["series"][2],
                            line=dict(width=2, color=P["bg"])),
                error_x=dict(type="data", symmetric=False, array=[r["high"] - r["predicted"]],
                             arrayminus=[r["predicted"] - r["low"]], color=P["muted"], thickness=2),
                showlegend=False, hovertemplate=f"{r['subject']}: {r['predicted']:.0f}%<extra></extra>"))
        fig.add_vline(x=40, line_dash="dot", line_color=P["bad"])
        fig.add_vline(x=70, line_dash="dot", line_color=P["good"])
        fig.update_layout(xaxis_range=[0, 100])
        theme.show(theme.style_fig(fig, 90 + 56 * len(results), legend=False), "adv_pred")

        table = pd.DataFrame([{"Subject": r["subject"], "CA avg": f"{r['ca_avg']:.0f}", "Trend": f"{r['trend']:+.0f}",
                               "Result": f"{r['predicted']:.0f}%", "Likely range": f"{r['low']:.0f}-{r['high']:.0f}%",
                               "Grade": r["grade"], "Type": r["status"], "Risk": r["risk"]} for r in results])
        theme.html_table(table, num_cols=("CA avg", "Trend", "Result", "Likely range"), tone_col="Risk", tones=RISK_TONE)

        st.subheader("Goal calculator")
        pending = [r for r in results if r["exam"] is None]
        if pending:
            g1, g2 = st.columns(2)
            pick = g1.selectbox("Subject", [r["subject"] for r in pending], key="adv_goal_subj")
            target = g2.selectbox("Target overall mark", [40, 50, 60, 70, 80, 90], index=2, key="adv_goal_t")
            r = next(x for x in pending if x["subject"] == pick)
            need = eng.needed_exam(r["ca_avg"], target)
            if need > 100:
                theme.callout(f"You would need {need:.0f}% in the exam. That is not reachable with your current coursework.", "bad")
            elif need <= 0:
                theme.callout(f"Already secured. Any exam mark gives you at least {target}%.", "good")
            else:
                theme.callout(f"To finish {pick} on {target}% you need {need:.0f}% in the exam "
                              f"(coursework average {r['ca_avg']:.0f}%).", "info")
        else:
            st.caption("All exams are entered, so there is nothing left to predict.")

# --------------------------------------------------------- recommendations
with t_rec:
    if not results:
        theme.callout(NO_DATA, "info")
    else:
        st.subheader("Personalised recommendations, most urgent first")
        for r in sorted(results, key=lambda r: -r["priority"]):
            with st.expander(f"{r['subject']}  |  {r['predicted']:.0f}%  |  Grade {r['grade']}  |  {r['risk']} risk",
                             expanded=r["weak"]):
                a, b = st.columns(2)
                with a:
                    st.markdown(theme.badge(f"{r['risk']} risk", RISK_TONE[r["risk"]]), unsafe_allow_html=True)
                    st.markdown("**Diagnosis**")
                    for x in r["reasons"]:
                        st.markdown(f"- {x}")
                    st.markdown("**Study activities**")
                    for x in r["activities"]:
                        st.markdown(f"- {x}")
                with b:
                    st.markdown("**Learning resources**")
                    for t, u in eng.resources_for(r["subject"]):
                        st.markdown(f"- [{t}]({u})" if u else f"- {t}")

        st.subheader("Weekly study plan")
        st.caption(f"Your {study_hours} study hours per week, shared in proportion to need.")
        plan = eng.study_plan(results, study_hours or 12)
        pf = go.Figure(go.Bar(x=[p["Hours / week"] for p in plan][::-1], y=[p["Subject"] for p in plan][::-1],
                              orientation="h", marker_color=P["series"][0], marker_cornerradius=4,
                              text=[f"{p['Hours / week']:.1f} h" for p in plan][::-1], textposition="outside",
                              textfont=dict(color=P["text"])))
        theme.show(theme.style_fig(pf, 80 + 46 * len(plan), legend=False), "adv_plan")
        pt = pd.DataFrame(plan)
        pt["Hours / week"] = pt["Hours / week"].map(lambda h: f"{h:.1f}")
        theme.html_table(pt, num_cols=("Hours / week",), tone_col="Priority", tones=RISK_TONE)

# -------------------------------------------------------------------- chat
with t_chat:
    st.subheader("Chat with your advisor")
    if not ss.adv_chat:
        ss.adv_chat.append(("assistant", eng.chat_response("hello", profile, results)))

    chips = ["Which subjects am I weak in?", "Predict my final marks", "Make me a study plan",
             "Recommend resources", "Help"]
    cols = st.columns(len(chips))
    clicked = None
    for col, s in zip(cols, chips):
        if col.button(s, key=f"adv_chip_{s}", width="stretch", icon=":material/chat_bubble:"):
            clicked = s

    for role, text in ss.adv_chat:
        with st.chat_message(role):
            st.markdown(text)

    typed = st.chat_input("Ask: 'How can I improve <subject>?' or 'What if I score 70 in <subject> exam?'",
                          key="adv_input")
    question = clicked or typed
    if question:
        ss.adv_chat.append(("user", question))
        ss.adv_chat.append(("assistant", eng.chat_response(question, profile, results)))
        st.rerun()

# ------------------------------------------------------------------ report
with t_rep:
    if not results:
        theme.callout(NO_DATA, "info")
    else:
        st.subheader("Personalised performance report")
        if not (name and reg_no and course):
            theme.callout("Fill in your name, registration number and course for a complete report.", "info")
        report = eng.generate_report_html(profile, results)
        st.download_button("Download report (HTML, print to PDF)", report, icon=":material/download:",
                           file_name=f"performance_report_{(reg_no or 'student').replace('/', '-')}.html",
                           mime="text/html", type="primary")
        try:
            st.iframe(report, height=700)
        except Exception:
            import streamlit.components.v1 as components
            components.html(report, height=700, scrolling=True)
