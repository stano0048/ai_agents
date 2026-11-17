"""Option 2 - AI-Powered Customer Support Agent (university student support) - page."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from support_engine import ANSWER_T, CLARIFY_T, SupportBot

P = theme.pal()
STATUS_TONE = {"answered": "good", "smalltalk": "muted", "clarify": "warn", "fallback": "bad", "escalate": "info"}
STATUS_TEXT = {"answered": "Answered", "smalltalk": "Small talk", "clarify": "Needs clarification",
               "fallback": "Could not answer", "escalate": "Escalated to staff"}

theme.hero("Student Support Agent",
           "Ask in your own words. The agent works out what you mean, finds the answer in its knowledge base, "
           "remembers the conversation and learns from your feedback.")

ss = st.session_state
if "sup_bot" not in ss:
    ss.sup_bot = SupportBot(Path(__file__).resolve().parent.parent / "learned_support.json")
    ss.sup_msgs = []          # {"role","text","meta"?,"fb"?}
    ss.sup_eval = None
bot: SupportBot = ss.sup_bot


# ------------------------------------------------------------- callbacks
def handle(text: str):
    text = text.strip()
    if not text:
        return
    hist = [{"role": m["role"], "text": m["text"]} for m in ss.sup_msgs]
    resp = bot.respond(text, hist)
    ss.sup_msgs.append({"role": "user", "text": text})
    ss.sup_msgs.append({"role": "assistant", "text": resp["text"], "meta": resp, "fb": 0})


def rate(idx: int, helpful: bool):
    m = ss.sup_msgs[idx]
    bot.feedback(m["meta"]["log_index"], helpful)
    m["fb"] = 1 if helpful else -1


def pick(idx: int, intent: str):
    """User chose the topic they really meant: learn it and show that answer."""
    m = ss.sup_msgs[idx]
    bot.correct(m["meta"]["log_index"], intent)
    m["fb"] = 1
    a = bot.answer_for(intent)
    ss.sup_msgs.append({"role": "assistant", "text": "Thanks, that helps me learn.\n\n" + a["text"],
                        "meta": {**m["meta"], "status": "answered", "intent": intent, "title": a["title"],
                                 "followups": a["followups"], "candidates": [], "learned": True}, "fb": 1})


def reset_chat():
    ss.sup_msgs = []
    bot.log.clear()


t_chat, t_stats, t_kb, t_how = st.tabs([":material/forum: Chat", ":material/bar_chart: Analytics",
                                         ":material/menu_book: Knowledge base", ":material/info: How it works"])

# ------------------------------------------------------------------- chat
with t_chat:
    left, right = st.columns([3, 2], gap="large")
    with left:
        chips = ["Pay fees", "Exam dates", "Reset password", "Hostels"]
        cols = st.columns(len(chips))
        for col, q in zip(cols, chips):
            col.button(q, key=f"sup_chip_{q}", icon=":material/help:", on_click=handle, args=(q,), width="stretch")

        box = st.container(height=470, border=True)
        with box:
            if not ss.sup_msgs:
                st.markdown("**Assistant**")
                st.markdown("Hello! Ask me about fees, registration, exams, results, accommodation, the library, "
                            "IT access or clearance. You can type naturally, for example: "
                            "*'my payment is not reflecting'*.")
            last_assistant = max((i for i, m in enumerate(ss.sup_msgs) if m["role"] == "assistant"), default=-1)
            for i, m in enumerate(ss.sup_msgs):
                with st.chat_message(m["role"]):
                    st.markdown(m["text"])
                    if m["role"] != "assistant":
                        continue
                    meta = m["meta"]
                    if meta["status"] == "clarify" or (meta["status"] == "fallback" and meta.get("candidates")):
                        st.caption("Pick the closest topic (the agent learns from your choice):")
                        for c in meta["candidates"][:3]:
                            if m["fb"] == 0:
                                st.button(c["title"], key=f"sup_pick_{i}_{c['intent']}", icon=":material/arrow_forward:",
                                          on_click=pick, args=(i, c["intent"]))
                    if m["fb"] == 0 and meta["status"] in ("answered", "clarify", "fallback"):
                        b1, b2, _ = st.columns([2, 2, 3])
                        b1.button("Helpful", key=f"sup_up_{i}", icon=":material/thumb_up:", on_click=rate, args=(i, True))
                        b2.button("Not helpful", key=f"sup_dn_{i}", icon=":material/thumb_down:", on_click=rate, args=(i, False))
                    elif m["fb"] != 0:
                        st.markdown(theme.badge("Feedback recorded: " + ("helpful" if m["fb"] > 0 else "not helpful"),
                                                "good" if m["fb"] > 0 else "warn"), unsafe_allow_html=True)
                    if i == last_assistant and meta.get("followups"):
                        for f in meta["followups"]:
                            st.button(f, key=f"sup_fu_{i}_{f}", icon=":material/subdirectory_arrow_right:",
                                      on_click=handle, args=(f,))
        q = st.chat_input("Type your question...", key="sup_input")
        if q:
            handle(q)
            st.rerun()
        st.button("Clear conversation", icon=":material/delete_sweep:", on_click=reset_chat, key="sup_clear")

    with right:
        st.subheader("Agent insight")
        st.caption("What the agent understood about your last message.")
        last = next((m["meta"] for m in reversed(ss.sup_msgs) if m["role"] == "assistant" and "log_index" in m["meta"]), None)
        if not last:
            theme.callout("Send a message to see how the agent reasons.", "info")
        else:
            matched = last["status"] != "fallback"
            theme.cards([("Detected intent", last["title"] if matched else "No reliable match", "", None)])
            st.write("")
            st.markdown(f"**Confidence {last['confidence'] * 100:.0f}%**  " + theme.badge(STATUS_TEXT[last["status"]],
                        STATUS_TONE[last["status"]]), unsafe_allow_html=True)
            st.markdown(theme.bar(last["confidence"] * 100), unsafe_allow_html=True)
            st.caption(f"Answer above {ANSWER_T:.0%}, ask to clarify above {CLARIFY_T:.0%}, otherwise fall back.")
            rows = [("Classifier probability", f"{last['classifier']:.2f}"), ("Retrieval similarity", f"{last['retrieval']:.2f}"),
                    ("Sentiment", last["sentiment"]), ("Used earlier context", "yes" if last["context_used"] else "no")]
            for k, v in last["entities"].items():
                rows.append((k.replace("_", " ").capitalize(), v))
            theme.html_table(pd.DataFrame(rows, columns=["Signal", "Value"]))
            if last["context_used"]:
                st.caption(f"Interpreted as: {last['query']}")
            if last["candidates"]:
                st.markdown("**Other topics considered**")
                theme.html_table(pd.DataFrame([{"Topic": c["title"], "Score": f"{c['score']:.2f}"}
                                               for c in last["candidates"]]), num_cols=("Score",))
            if last.get("ticket"):
                theme.callout(f"Ticket {last['ticket']['id']} created with the last messages attached.", "info")

# --------------------------------------------------------------- analytics
with t_stats:
    s = bot.stats()
    learned = len(bot.extra)
    if s["n"] == 0:
        theme.callout("No conversations yet. Ask the agent something in the Chat tab.", "info")
    else:
        sat = s["satisfaction"]
        theme.cards([("Questions", str(s["n"]), "this session"),
                     ("Answered", f"{s['answered'] / s['n']:.0%}", f"{s['answered']} of {s['n']}", "good"),
                     ("Could not answer", str(s["fallback"]), "fallbacks", "bad" if s["fallback"] else None),
                     ("Satisfaction", "-" if sat is None else f"{sat:.0%}", "from thumbs up/down"),
                     ("Learned phrases", str(learned), "added from feedback")])
        st.write("")
        df = s["df"]
        a, b = st.columns(2)
        with a:
            counts = df[df["status"] != "fallback"]["intent"].value_counts().head(8)
            fig = go.Figure(go.Bar(y=[bot.article(i)["title"] for i in counts.index][::-1], x=counts.values[::-1],
                                   orientation="h", marker_color=P["series"][0], marker_cornerradius=4))
            fig.update_layout(title="Most asked topics", xaxis=dict(dtick=1))
            theme.show(theme.style_fig(fig, 340, legend=False), "sup_topics")
        with b:
            fig = go.Figure(go.Scatter(x=list(range(1, len(df) + 1)), y=df["confidence"], mode="lines+markers",
                                       line=dict(color=P["series"][0], width=2),
                                       marker=dict(size=8, line=dict(width=2, color=P["bg"]))))
            fig.add_hline(y=ANSWER_T, line_dash="dot", line_color=P["good"], annotation_text="Answer threshold",
                          annotation_font_color=P["muted"])
            fig.update_layout(title="Confidence per question", yaxis_range=[0, 1], xaxis_title="Question number")
            theme.show(theme.style_fig(fig, 340, legend=False), "sup_conf")
        log = df.tail(12).iloc[::-1][["text", "intent", "confidence", "status", "feedback"]].copy()
        log["intent"] = log["intent"].map(lambda i: bot.article(i)["title"])
        log["confidence"] = log["confidence"].map(lambda c: f"{c:.2f}")
        log["feedback"] = log["feedback"].map({1: "helpful", -1: "not helpful", 0: "-"})
        log.columns = ["Question", "Detected intent", "Confidence", "Outcome", "Feedback"]
        st.markdown("**Recent questions**")
        theme.html_table(log, num_cols=("Confidence",), tone_col="Outcome",
                         tones={"answered": "good", "smalltalk": "muted", "clarify": "warn", "fallback": "bad", "escalate": "info"})
    if bot.penalty:
        st.caption("Topics the agent now treats with extra caution after negative feedback: " +
                   ", ".join(f"{bot.article(k)['title']} ({v:.2f})" for k, v in bot.penalty.items() if v > 0) or "none")

# ---------------------------------------------------------- knowledge base
with t_kb:
    st.subheader("Knowledge base")
    st.caption("Sample content for demonstration. Replace the answers with your institution's official information.")
    query = st.text_input("Filter topics", key="sup_kb_filter", placeholder="Search by title or category")
    rows = [{"Topic": a["title"], "Category": a["category"], "Training phrases": len(a["examples"]) +
             sum(1 for _, i in bot.extra if i == a["id"]), "Source": "Added by you" if a.get("custom") else "Built in"}
            for a in bot.kb if a["id"] != "out_of_scope" and
            (not query or query.lower() in (a["title"] + a["category"]).lower())]
    theme.html_table(pd.DataFrame(rows), num_cols=("Training phrases",))

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Teach the agent a new topic**")
        with st.form("sup_add", clear_on_submit=True):
            title = st.text_input("Topic title")
            answer = st.text_area("Answer (markdown allowed)")
            ex = st.text_area("Example questions, one per line", help="Provide at least three different phrasings.")
            ok = st.form_submit_button("Add topic and retrain", icon=":material/add:", type="primary")
        if ok:
            lines = [x.strip() for x in ex.splitlines() if x.strip()]
            if not title.strip() or not answer.strip() or len(lines) < 3:
                theme.callout("Please give a title, an answer and at least three example questions.", "warn")
            else:
                bot.add_article(title.strip(), answer.strip(), lines)
                theme.callout("Topic added and the intent classifier was retrained.", "good")
    with right:
        st.markdown("**Classifier playground**")
        phrase = st.text_input("Try a phrase", key="sup_play", placeholder="e.g. my deposit is not showing")
        if phrase:
            r = bot.score(phrase)[:5]
            theme.html_table(pd.DataFrame([{"Topic": bot.article(i)["title"], "Score": f"{sc:.2f}",
                                            "Classifier": f"{p:.2f}", "Retrieval": f"{c:.2f}"} for i, sc, p, c in r]),
                             num_cols=("Score", "Classifier", "Retrieval"))

# ------------------------------------------------------------ how it works
with t_how:
    st.subheader("How the agent works")
    st.markdown("""
1. **Understanding (NLP).** Each message is normalised (lower-case, synonyms such as *fees/tuition*), and entities such as
   registration numbers, dates and amounts are extracted. Frustration words are detected so the agent can apologise and offer a human.
2. **Intent classification.** A Logistic Regression model on TF-IDF word and character n-grams predicts the topic. Character
   n-grams make it tolerant of typing mistakes.
3. **Information retrieval.** TF-IDF cosine similarity searches the knowledge base independently of the classifier.
4. **Decision.** The two scores are blended. High confidence gives an answer, medium confidence asks *"did you mean...?"*,
   and low confidence admits it does not know instead of guessing.
5. **Memory.** The whole conversation is kept. A short follow-up such as *"and the deadline?"* is combined with the
   previous question.
6. **Learning.** Thumbs up adds your wording as a new training example, thumbs down lowers that topic's score, and picking the
   right topic adds a corrected example. The classifier retrains instantly and the learning is saved to `learned_support.json`.
""")
    if st.button("Measure classifier accuracy (cross-validation)", icon=":material/speed:", key="sup_eval_btn"):
        ss.sup_eval = bot.evaluate()
    if ss.sup_eval:
        e = ss.sup_eval
        theme.cards([("Cross-validated accuracy", "-" if e["accuracy"] is None else f"{e['accuracy']:.0%}",
                      "unseen phrasings of known topics"),
                     ("Training phrases", str(e["n"]), ""), ("Topics", str(e["intents"]), "")])
    if st.button("Forget everything learned from feedback", icon=":material/restart_alt:", key="sup_reset"):
        bot.extra.clear(); bot.penalty.clear(); bot.train(); bot.save()
        st.rerun()
