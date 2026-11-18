"""Option 2 - AI-Powered Customer Support Agent (supermarket) - page."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import services
import theme
from support_engine import ANSWER_T, CLARIFY_T, SAMPLE_INBOX, SupportBot, triage_run

P = theme.pal()
STATUS_TONE = {"answered": "good", "smalltalk": "muted", "clarify": "warn", "fallback": "bad", "escalate": "info",
               "catalogue": "good", "researched": "info", "learned": "good"}
STATUS_TEXT = {"answered": "Answered", "smalltalk": "Small talk", "clarify": "Needs clarification",
               "fallback": "Could not answer", "escalate": "Escalated to staff", "catalogue": "From product catalogue",
               "researched": "Researched", "learned": "Remembered answer"}
RATEABLE = ("answered", "clarify", "fallback", "researched", "catalogue", "learned")

theme.hero("Supermarket Support Agent",
           "Ask in your own words. The agent checks its knowledge base and the live product catalogue, and when neither has "
           "the answer it researches the question and cites its sources. It remembers what it learns.")
services.source_button("support")

ss = st.session_state
if "sup_bot" not in ss:
    ss.sup_bot = SupportBot(services.database(), services.researcher())
    ss.sup_msgs = []          # {"role","text","meta"?,"fb"?}
    ss.sup_eval = None
bot: SupportBot = ss.sup_bot
bot.researcher = services.researcher()       # picks up a Gemini key pasted after the page loaded


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


t_chat, t_tri, t_stats, t_kb, t_eval = st.tabs([":material/forum: Chat", ":material/inbox: Inbox triage",
                                              ":material/bar_chart: Analytics", ":material/menu_book: Knowledge and catalogue",
                                              ":material/speed: Accuracy"])

# ------------------------------------------------------------------- chat
with t_chat:
    left, right = st.columns([3, 2], gap="large")
    with left:
        chips = {"Offers": "What offers do you have?", "Delivery": "Do you deliver?", "Sugar price": "How much is sugar?",
                 "Rice storage": "How long can I keep cooked rice in the fridge?"}
        cols = st.columns(len(chips))
        for col, (label, q) in zip(cols, chips.items()):
            col.button(label, key=f"sup_chip_{label}", icon=":material/help:", on_click=handle, args=(q,), width="stretch")

        box = st.container(height=470, border=True)
        with box:
            if not ss.sup_msgs:
                st.markdown("**Assistant**")
                st.markdown("Hello! Ask me about prices, stock, offers, delivery, payments, loyalty points or returns. "
                            "If I do not know, I will look it up and show where the answer came from. "
                            "Try: *'how long does milk last once opened?'*")
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
                    if meta.get("sources"):
                        st.caption("Sources: " + " | ".join(f"[{x['title'][:40]}]({x['url']})" for x in meta["sources"][:4]))
                    if meta.get("status") == "researched":
                        st.caption("Researched " + ("with Gemini and Google Search" if meta.get("mode") == "gemini" else
                                                    "from Wikipedia") + ", not from the store's own records. "
                                   "Confirm store-specific details with the team.")
                    if m["fb"] == 0 and meta["status"] in RATEABLE:
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
            matched = last["status"] not in ("fallback", "researched")
            theme.cards([("Detected intent", last["title"] if matched else ("Open question" if last["status"] == "researched" else "No reliable match"), "", None)])
            st.write("")
            st.markdown(f"**Confidence {last['confidence'] * 100:.0f}%**  " + theme.badge(STATUS_TEXT[last["status"]],
                        STATUS_TONE[last["status"]]), unsafe_allow_html=True)
            st.markdown(theme.bar(last["confidence"] * 100), unsafe_allow_html=True)
            st.caption(f"Answer above {ANSWER_T:.0%}, ask to clarify above {CLARIFY_T:.0%}, otherwise fall back.")
            rows = [("Classifier probability", f"{last['classifier']:.2f}"), ("Retrieval similarity", f"{last['retrieval']:.2f}"),
                    ("Sentiment", last["sentiment"]), ("Used earlier context", "yes" if last["context_used"] else "no"),
                    ("Answer source", {"gemini": "Gemini + Google Search", "live": "Wikipedia", "catalogue": "Product catalogue",
                                       "learned": "Learned answer"}.get(last.get("mode"), "Knowledge base"))]
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

# ------------------------------------------------------------ inbox triage
with t_tri:
    st.subheader("Autonomous inbox triage")
    st.caption("Paste several customer messages. The agent reads them all, answers what it can, ranks urgency, routes the "
               "rest to the right office and opens tickets, with no human in the loop.")
    ss.setdefault("sup_inbox", "\n".join(SAMPLE_INBOX))
    st.text_area("Incoming messages, one per line", key="sup_inbox", height=210)
    if st.button("Process inbox autonomously", icon=":material/play_arrow:", type="primary", key="sup_tri_btn"):
        ss.sup_triage = bot.triage(ss.sup_inbox.splitlines())
    rows = ss.get("sup_triage")
    if rows:
        auto = sum(r["status"] in ("answered", "smalltalk", "catalogue", "researched", "learned") for r in rows)
        flagged = sum(r["priority"] == "High" for r in rows)
        unans = sum(r["status"] == "fallback" for r in rows)
        theme.cards([("Messages", str(len(rows)), "processed"), ("Answered automatically", str(auto), "no staff time", "good"),
                     ("High priority", str(flagged), "urgent or frustrated", "bad" if flagged else None),
                     ("Needs a person", str(unans + sum(r["status"] == "escalate" for r in rows)), "escalated or unanswered")])
        st.write("")
        st.markdown("**Agent cycle for this batch**")
        theme.agent_loop(triage_run(rows))
        st.markdown("**Decision for every message**")
        tbl = pd.DataFrame([{"Message": r["message"][:58] + ("..." if len(r["message"]) > 58 else ""), "Topic": r["intent"],
                             "Sure": f"{r['confidence']:.0%}", "Action taken": r["action"], "Office": r["department"],
                             "Priority": r["priority"], "Ticket": r["ticket"]} for r in rows])
        theme.html_table(tbl, num_cols=("Sure",), tone_col="Priority", tones={"High": "bad", "Normal": "muted"})
        with st.expander("See the replies the agent sent"):
            for r in rows:
                if r["status"] in ("answered", "smalltalk", "catalogue", "researched", "learned"):
                    st.markdown(f"**{r['message']}**")
                    st.markdown(r["reply"])
                    st.divider()
        if bot.tickets:
            st.markdown("**Ticket queue**")
            theme.html_table(pd.DataFrame([{"Ticket": t["id"], "Office": t["department"], "Priority": t["priority"],
                                            "Question": t["question"][:70]} for t in bot.tickets]),
                             tone_col="Priority", tones={"High": "bad", "Normal": "muted"})
            st.caption("Tickets are stored in the database. In a real deployment they would also be sent to the department's helpdesk or email.")

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
                     ("Researched", str(s["researched"]), "open questions", "info" if s["researched"] else None),
                     ("Could not answer", str(s["fallback"]), "fallbacks", "bad" if s["fallback"] else None),
                     ("Satisfaction", "-" if sat is None else f"{sat:.0%}", "from thumbs up/down"),
                     ("Learned phrases", str(learned), "added from feedback")])
        st.write("")
        df = s["df"]
        a, b = st.columns(2)
        with a:
            counts = df[~df["status"].isin(["fallback", "researched"])]["intent"].value_counts().head(8)
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
                         tones=STATUS_TONE)
    if bot.penalty:
        st.caption("Topics the agent now treats with extra caution after negative feedback: " +
                   ", ".join(f"{bot.article(k)['title']} ({v:.2f})" for k, v in bot.penalty.items() if v > 0) or "none")

# ---------------------------------------------------------- knowledge base
with t_kb:
    st.subheader("Knowledge base")
    st.caption("Sample policies for a fictional store. Replace the answers with the real store's information.")
    query = st.text_input("Filter topics", key="sup_kb_filter", placeholder="Search by title or category")
    rows = [{"Topic": a["title"], "Category": a["category"], "Training phrases": len(a["examples"]) +
             sum(1 for _, i in bot.extra if i == a["id"]), "Source": "Added by staff" if a.get("custom") else "Built in"}
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
        phrase = st.text_input("Try a phrase", key="sup_play", placeholder="e.g. can I pay with my card")
        if phrase:
            r = bot.score(phrase)[:5]
            theme.html_table(pd.DataFrame([{"Topic": bot.article(i)["title"], "Score": f"{sc:.2f}",
                                            "Classifier": f"{p:.2f}", "Retrieval": f"{c:.2f}"} for i, sc, p, c in r]),
                             num_cols=("Score", "Classifier", "Retrieval"))

    st.markdown("**Product catalogue (live from the database)**")
    prods = pd.DataFrame(bot.products())
    if not prods.empty:
        prods["Offer"] = prods["offer_price"].map(lambda v: "-" if v != v or v is None else f"{v:,.0f}")
        prods["Stock"] = prods["stock"].map(lambda v: "Out of stock" if v <= 0 else str(v))
        view = prods[["name", "category", "unit", "price", "Offer", "Stock", "aisle"]].copy()
        view["price"] = view["price"].map(lambda v: f"{v:,.0f}")
        view.columns = ["Product", "Category", "Unit", "Price (KES)", "Offer (KES)", "Stock", "Aisle"]
        flt = st.text_input("Filter products", key="sup_prod_filter", placeholder="e.g. milk or dairy")
        if flt:
            view = view[view.apply(lambda r: flt.lower() in " ".join(map(str, r)).lower(), axis=1)]
        theme.html_table(view.head(60), num_cols=("Price (KES)", "Offer (KES)"), tone_col="Stock", tones={"Out of stock": "bad"})
        st.caption("Sample products and prices. The agent reads this table when a customer asks about price, stock or aisle.")

# ------------------------------------------------------------------- accuracy
with t_eval:
    st.subheader("How well does it understand?")
    st.caption("Measured on the built-in test phrasings. The held-out phrasings were written after the training phrases and the "
               "agent was not tuned to them.")
    if st.button("Measure accuracy", icon=":material/speed:", key="sup_eval_btn"):
        ss.sup_eval = {"cv": bot.evaluate(), "held": bot.evaluate_heldout()}
    if ss.sup_eval:
        cv, held = ss.sup_eval["cv"], ss.sup_eval["held"]
        theme.cards([("Held-out accuracy", f"{held['accuracy']:.0%}", f"{held['n']} unseen phrasings"),
                     ("Cross-validated accuracy", "-" if cv["accuracy"] is None else f"{cv['accuracy']:.0%}", "known topics"),
                     ("Training phrases", str(cv["n"]), f"{cv['intents']} topics")])
        if held["wrong"]:
            st.markdown("**Where it was wrong**")
            theme.html_table(pd.DataFrame([{"Phrase": t, "Expected": e, "Predicted": g} for t, e, g in held["wrong"]]))
    if st.button("Forget everything learned from feedback", icon=":material/restart_alt:", key="sup_reset"):
        bot.reset_learning()
        st.rerun()
