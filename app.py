"""
AI-Powered Student Academic Advisory Agent  -  Streamlit interface
Run with:   streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import advisor_engine as eng

st.set_page_config(page_title="AI Academic Advisor", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#4338ca 0%,#7c3aed 55%,#db2777 100%);
      padding:26px 30px;border-radius:18px;color:#fff;margin-bottom:18px}
.hero h1{margin:0;font-size:2rem;color:#fff} .hero p{margin:4px 0 0;opacity:.92}
.kpi{border-radius:16px;padding:16px 18px;color:#fff;text-align:center;
     box-shadow:0 4px 14px rgba(0,0,0,.12)}
.kpi .v{font-size:2rem;font-weight:700;line-height:1.1} .kpi .l{font-size:.85rem;opacity:.9}
.k1{background:linear-gradient(135deg,#4f46e5,#6366f1)} .k2{background:linear-gradient(135deg,#0ea5e9,#2563eb)}
.k3{background:linear-gradient(135deg,#f59e0b,#ef4444)} .k4{background:linear-gradient(135deg,#10b981,#059669)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:600;color:#fff}
.High{background:#e5484d}.Medium{background:#f5a524}.Watch{background:#c9a800}.Low{background:#2fb344}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🎓 AI Academic Advisor</h1>
<p>Analyses your marks, predicts your final results, diagnoses weak subjects and coaches you with a personalised plan.</p></div>""",
            unsafe_allow_html=True)

# ------------------------------------------------------------ session state
if "marks" not in st.session_state:
    st.session_state.marks = pd.DataFrame(
        {"Subject": [""], "CAT1": [None], "CAT2": [None], "Assignment": [None], "Exam": [None]}
    ).astype({"CAT1": float, "CAT2": float, "Assignment": float, "Exam": float})
if "chat" not in st.session_state:
    st.session_state.chat = []

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("👤 Student details")
    name = st.text_input("Full name")
    reg_no = st.text_input("Registration number")
    course = st.text_input("Course / programme")
    st.divider()
    st.subheader("Study habits")
    attendance = st.slider("Class attendance (%)", 0, 100, 80)
    study_hours = st.slider("Study hours per week", 0, 40, 12)
    st.divider()
    if st.button("📥 Load sample data", width="stretch"):
        st.session_state.marks = eng.sample_marks().astype(
            {"CAT1": float, "CAT2": float, "Assignment": float, "Exam": float})
        st.session_state.chat = []
        st.rerun()
    if st.button("🗑️ Clear marks", width="stretch"):
        st.session_state.marks = st.session_state.marks.iloc[0:0]
        st.session_state.chat = []
        st.rerun()
    st.caption("Marks are percentages (0-100). CA = CATs + assignment (30%). Leave **Exam** blank "
               "if you haven't sat it - the AI will predict it.")

profile = {"name": name, "reg_no": reg_no, "course": course,
           "attendance": float(attendance), "study_hours": float(study_hours)}

# ------------------------------------------------------------------- tabs
tab_marks, tab_dash, tab_pred, tab_rec, tab_chat, tab_report = st.tabs(
    ["📝 Marks", "📊 Dashboard", "🔮 Predictions", "💡 Recommendations", "💬 Ask the Advisor", "📄 Report"])

with tab_marks:
    st.subheader("Enter your academic marks")
    st.caption("Click a cell to edit. Use the ➕ row at the bottom to add subjects.")
    edited = st.data_editor(
        st.session_state.marks, num_rows="dynamic", width="stretch", key="editor",
        column_config={
            "Subject": st.column_config.TextColumn("Subject", required=True),
            "CAT1": st.column_config.NumberColumn("CAT 1 (%)", min_value=0, max_value=100),
            "CAT2": st.column_config.NumberColumn("CAT 2 (%)", min_value=0, max_value=100),
            "Assignment": st.column_config.NumberColumn("Assignment (%)", min_value=0, max_value=100),
            "Exam": st.column_config.NumberColumn("Exam (%)", min_value=0, max_value=100),
        })
    st.session_state.marks = edited

results = eng.analyse(edited, attendance, study_hours)
summ = eng.overall_summary(results)

NO_DATA = "Add at least one subject with marks in the **Marks** tab (or click *Load sample data*)."

# ---------------------------------------------------------------- dashboard
with tab_dash:
    if not results:
        st.info(NO_DATA)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="kpi k1"><div class="v">{summ["average"]:.0f}%</div>'
                    f'<div class="l">Projected average</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi k2"><div class="v" style="font-size:1.15rem;padding:9px 0">'
                    f'{summ["classification"].split(" (")[0]}</div><div class="l">Projected standing</div></div>',
                    unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi k3"><div class="v">{summ["n_weak"]}/{summ["n_subjects"]}</div>'
                    f'<div class="l">Weak subjects</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi k4"><div class="v" style="font-size:1.15rem;padding:9px 0">'
                    f'{summ["best"]["subject"][:22]}</div><div class="l">Strongest subject</div></div>',
                    unsafe_allow_html=True)
        st.write("")

        left, right = st.columns([3, 2])
        with left:
            names = [r["subject"] for r in results]
            fig = go.Figure()
            fig.add_bar(x=names, y=[r["ca_avg"] for r in results], name="Coursework avg",
                        marker_color="#a5b4fc")
            fig.add_bar(x=names, y=[r["predicted"] for r in results], name="Final / predicted",
                        marker_color=["#e5484d" if r["risk"] == "High" else "#f5a524" if r["risk"] == "Medium"
                                      else "#6366f1" for r in results])
            for y, c, t in [(40, "#e5484d", "Pass 40"), (70, "#2fb344", "A 70")]:
                fig.add_hline(y=y, line_dash="dot", line_color=c, annotation_text=t)
            fig.update_layout(title="Performance by subject", barmode="group", yaxis_range=[0, 100],
                              height=380, margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")
        with right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=summ["average"], number={"suffix": "%"},
                title={"text": "Overall projection"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4338ca"},
                       "steps": [{"range": [0, 40], "color": "#fecaca"}, {"range": [40, 50], "color": "#fde68a"},
                                 {"range": [50, 60], "color": "#fef3c7"}, {"range": [60, 70], "color": "#bbf7d0"},
                                 {"range": [70, 100], "color": "#86efac"}]}))
            gauge.update_layout(height=380, margin=dict(t=60, b=10))
            st.plotly_chart(gauge, width="stretch")

        if len(results) >= 3:
            labels = [r["subject"] for r in results]
            vals = [r["predicted"] for r in results]
            radar = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself",
                                              line_color="#7c3aed", name="Predicted"))
            radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=380,
                                title="Strengths & weaknesses radar", margin=dict(t=60, b=20), showlegend=False)
            st.plotly_chart(radar, width="stretch")

        weak = [r for r in results if r["weak"]]
        if weak:
            st.warning("⚠️ **Weak subjects detected:** " + ", ".join(
                f"{r['subject']} ({r['predicted']:.0f}%)" for r in sorted(weak, key=lambda r: -r["priority"])))
        else:
            st.success("✅ No weak subjects detected. Keep it up!")

# -------------------------------------------------------------- predictions
with tab_pred:
    if not results:
        st.info(NO_DATA)
    else:
        st.subheader("Predicted final marks")
        st.caption("A regression model uses your CA average, mark trend, attendance and study hours. "
                   "The bar shows the likely (80%) range.")
        fig = go.Figure()
        for r in results:
            fig.add_trace(go.Scatter(
                x=[r["predicted"]], y=[r["subject"]], mode="markers",
                marker=dict(size=14, color="#6366f1" if r["status"] == "Predicted" else "#10b981"),
                error_x=dict(type="data", symmetric=False,
                             array=[r["high"] - r["predicted"]], arrayminus=[r["predicted"] - r["low"]],
                             color="#a5b4fc", thickness=4),
                showlegend=False, hovertemplate=f"{r['subject']}: {r['predicted']:.0f}%<extra></extra>"))
        fig.add_vline(x=40, line_dash="dot", line_color="#e5484d")
        fig.add_vline(x=70, line_dash="dot", line_color="#2fb344")
        fig.update_layout(xaxis_range=[0, 100], height=60 + 60 * len(results), margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        table = pd.DataFrame([{
            "Subject": r["subject"], "CA avg": round(r["ca_avg"]), "Trend": f"{r['trend']:+.0f}",
            "Result": f"{r['predicted']:.0f}%", "Range": f"{r['low']:.0f}-{r['high']:.0f}%",
            "Grade": r["grade"], "Type": r["status"], "Risk": r["risk"]} for r in results])
        st.dataframe(table, width="stretch", hide_index=True)

        st.subheader("🎯 Goal calculator")
        pending = [r for r in results if r["exam"] is None]
        if pending:
            g1, g2 = st.columns(2)
            pick = g1.selectbox("Subject", [r["subject"] for r in pending])
            target = g2.select_slider("Target overall mark", options=[40, 50, 60, 70, 80, 90], value=60)
            r = next(x for x in pending if x["subject"] == pick)
            need = eng.needed_exam(r["ca_avg"], target)
            if need > 100:
                st.error(f"You would need {need:.0f}% in the exam - not reachable with current coursework.")
            elif need <= 0:
                st.success(f"Already secured - any exam mark gives you {target}%.")
            else:
                st.info(f"To finish **{pick}** on **{target}%** you need **{need:.0f}%** in the exam "
                        f"(coursework avg {r['ca_avg']:.0f}%).")
        else:
            st.caption("All exams already entered - nothing left to predict.")

# ---------------------------------------------------------- recommendations
with tab_rec:
    if not results:
        st.info(NO_DATA)
    else:
        st.subheader("Personalised recommendations (most urgent first)")
        for r in sorted(results, key=lambda r: -r["priority"]):
            head = f"{r['subject']} - {r['predicted']:.0f}% (Grade {r['grade']})"
            with st.expander(head, expanded=r["weak"]):
                st.markdown(f'<span class="badge {r["risk"]}">{r["risk"]} risk</span>', unsafe_allow_html=True)
                a, b = st.columns(2)
                with a:
                    st.markdown("**🔎 AI diagnosis**")
                    for x in r["reasons"]:
                        st.markdown(f"- {x}")
                    st.markdown("**✅ Study activities**")
                    for x in r["activities"]:
                        st.markdown(f"- {x}")
                with b:
                    st.markdown("**📚 Learning resources**")
                    for t, u in eng.resources_for(r["subject"]):
                        st.markdown(f"- [{t}]({u})" if u else f"- {t}")

        st.subheader("🗓️ Suggested weekly study plan")
        plan = eng.study_plan(results, study_hours or 12)
        pc1, pc2 = st.columns([2, 3])
        with pc1:
            donut = go.Figure(go.Pie(labels=[p["Subject"] for p in plan],
                                     values=[p["Hours / week"] for p in plan], hole=.55))
            donut.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(donut, width="stretch")
        with pc2:
            st.dataframe(pd.DataFrame(plan), width="stretch", hide_index=True)

# --------------------------------------------------------------------- chat
with tab_chat:
    st.subheader("Chat with your advisor")
    if not st.session_state.chat:
        st.session_state.chat.append(("assistant", eng.chat_response("hello", profile, results)))

    suggestions = ["Which subjects am I weak in?", "Predict my final marks",
                   "Make me a study plan", "Recommend resources", "help"]
    cols = st.columns(len(suggestions))
    clicked = None
    for col, s in zip(cols, suggestions):
        if col.button(s, width="stretch", key=f"sg_{s}"):
            clicked = s

    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)

    typed = st.chat_input("Ask e.g. 'How can I improve Computer Security?' or 'What if I score 70 in exam?'")
    question = clicked or typed
    if question:
        answer = eng.chat_response(question, profile, results)
        st.session_state.chat.append(("user", question))
        st.session_state.chat.append(("assistant", answer))
        st.rerun()

# ------------------------------------------------------------------- report
with tab_report:
    if not results:
        st.info(NO_DATA)
    else:
        st.subheader("Personalised performance report")
        if not (name and reg_no and course):
            st.caption("Tip: fill in your name, registration number and course in the sidebar for a complete report.")
        report = eng.generate_report_html(profile, results)
        st.download_button("⬇️ Download report (HTML - open & print to PDF)", report,
                           file_name=f"performance_report_{(reg_no or 'student').replace('/', '-')}.html",
                           mime="text/html", type="primary")
        try:                                   # newer Streamlit
            st.iframe(report, height=700)
        except Exception:                      # older Streamlit fallback
            import streamlit.components.v1 as components
            components.html(report, height=700, scrolling=True)
