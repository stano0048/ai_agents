"""
report.py - writes the group report as a PDF, following the report outline in the assignment handout (section 5):
introduction, problem statements, objectives, agent and environment, design, AI techniques, source code, screenshots,
testing, challenges, conclusion; plus group participation, assessment mapping and deliverables.

Everything that can change (members, responsibilities, test results, metrics) is read from the database at the moment the
report is built, so the PDF is always current.
"""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent
ACCENT, INK, MUTED, LINE, SOFT = "#2a78d6", "#1c2333", "#5b6475", "#d8d7d2", "#f1f1ee"
UNIVERSITY = "Karatina University"
UNIT = "Python Programming: Artificial Intelligence and Intelligent Agents"

ST = {
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14.5, textColor=colors.HexColor(INK), spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=colors.HexColor(MUTED)),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.8, leading=11.5, textColor=colors.HexColor(INK)),
    "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=8.8, leading=11.5, textColor=colors.white),
    "H1": ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=colors.HexColor(INK), spaceBefore=4, spaceAfter=8),
    "H2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=colors.HexColor(ACCENT), spaceBefore=8, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=10, leading=14, leftIndent=14, bulletIndent=2, spaceAfter=2,
                             textColor=colors.HexColor(INK)),
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26, leading=31, textColor=colors.HexColor(INK)),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=12.5, leading=17, textColor=colors.HexColor(MUTED)),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.8, leading=10, textColor=colors.HexColor(INK), backColor=colors.HexColor(SOFT),
                           borderPadding=6, leftIndent=4, spaceAfter=8),
}


def P(t, s="body"):
    return Paragraph(t, ST[s])


def bullets(items):
    return [Paragraph(i, ST["bullet"], bulletText="•") for i in items]


def table(rows, widths, header=True, align_right=()):
    data = [[Paragraph(str(c), ST["cellh" if (header and r == 0) else "cell"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(LINE)),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT))]
        style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor(SOFT)) for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


class Doc(BaseDocTemplate):
    def __init__(self, buf, **kw):
        super().__init__(buf, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.2 * cm, bottomMargin=2.0 * cm,
                         title="Group Report: AI Agents Suite", author=UNIVERSITY, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._page)])

    def _page(self, canv, doc):
        canv.saveState()
        if doc.page > 1:
            canv.setFont("Helvetica", 8)
            canv.setFillColor(colors.HexColor(MUTED))
            canv.drawString(self.leftMargin, A4[1] - 1.3 * cm, "AI Agents Suite - Group report")
            canv.drawRightString(A4[0] - self.rightMargin, 1.2 * cm, f"Page {doc.page}")
            canv.setStrokeColor(colors.HexColor(LINE))
            canv.line(self.leftMargin, A4[1] - 1.5 * cm, A4[0] - self.rightMargin, A4[1] - 1.5 * cm)
        canv.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name == "H1":
            self.notify("TOCEntry", (0, fl.getPlainText(), self.page))


def architecture_png() -> bytes:
    """Layered architecture drawn with NetworkX + Matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    layers = {
        "ui": (["Streamlit pages\n(Overview, Advisor, Support,\nFinance, Group, Testing)", "Command line\n(cli.py)"], 3.0),
        "agents": (["Academic Advisor\nengine", "Support Agent\nengine", "Finance Agent\nengine"], 2.0),
        "services": (["Researcher\n(research.py)", "Gemini client\n(llm.py)", "Database layer\n(db.py)"], 1.0),
        "external": (["Gemini API +\nGoogle Search", "Wikipedia +\nOpen Library", "SQLite file\n(agents.db)"], 0.0),
    }
    pos, colour = {}, {}
    fills = {"ui": "#d6e6fa", "agents": "#2a78d6", "services": "#e8e8e4", "external": "#fbe9c8"}
    for layer, (names, y) in layers.items():
        n = len(names)
        for i, nm in enumerate(names):
            G.add_node(nm)
            pos[nm] = ((i + 0.5) / n * 10, y * 2.2)
            colour[nm] = fills[layer]
    ui, ag, sv, ex = (layers[k][0] for k in ("ui", "agents", "services", "external"))
    edges = [(ui[0], a) for a in ag] + [(ui[1], a) for a in ag] + [(ag[0], sv[0]), (ag[1], sv[0]), (ag[1], sv[2]), (ag[2], sv[1]),
             (ag[2], sv[2]), (ag[0], sv[2]), (sv[0], sv[1]), (sv[0], ex[1]), (sv[1], ex[0]), (sv[2], ex[2])]
    G.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#8a93a6", arrows=True, arrowsize=12, width=1.2, node_size=5200, node_shape="s")
    for n_, (x, y) in pos.items():
        txt_col = "white" if colour[n_] == "#2a78d6" else "#1c2333"
        ax.text(x, y, n_, ha="center", va="center", fontsize=8.5, color=txt_col, zorder=5,
                bbox=dict(boxstyle="round,pad=0.55", fc=colour[n_], ec="#8a93a6", lw=0.8))
    for label, y in [("Interface", 3.0), ("Agents", 2.0), ("Services", 1.0), ("External and storage", 0.0)]:
        ax.text(-0.3, y * 2.2, label, rotation=90, va="center", ha="center", fontsize=9, color="#5b6475")
    ax.set_xlim(-0.7, 10.3); ax.set_ylim(-1.0, 7.6); ax.axis("off")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return buf.getvalue()


def _img(path_or_bytes, max_w, max_h):
    from PIL import Image as PILImage
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else str(path_or_bytes)
    w, h = PILImage.open(src).size
    if isinstance(src, io.BytesIO):
        src.seek(0)
    k = min(max_w / w, max_h / h)
    return Image(src, width=w * k, height=h * k)


def _loc() -> list[tuple[str, int, str]]:
    purposes = {"advisor_engine.py": "Academic Advisor: prediction, rules, planning", "advisor_chat.py": "Advisor conversation: intent understanding", "support_engine.py": "Support Agent: NLP, catalogue, learning",
                "finance_engine.py": "Finance Agent: categoriser, patterns, forecast", "research.py": "Research layer (Gemini, Wikipedia, offline)",
                "llm.py": "Gemini REST client", "db.py": "SQLite database layer", "report.py": "This PDF report", "qa_tests.py": "Automated tests",
                "theme.py": "Light/dark design system", "services.py": "Shared Streamlit services", "sourcezip.py": "Source-code downloads",
                "datasets.py": "Sample datasets", "cli.py": "Command-line interface", "app.py": "App router",
                "views/advisor.py": "Advisor page", "views/support.py": "Support page", "views/finance.py": "Finance page",
                "views/group.py": "Group members page", "views/testing.py": "Testing page", "views/home.py": "Overview page"}
    out = []
    for f, d in purposes.items():
        p = ROOT / f
        if p.exists():
            out.append((f, len(p.read_text(encoding="utf-8").splitlines()), d))
    return out


def build_pdf(db) -> bytes:
    import qa_tests
    members = db.list_members()
    tests = db.load_tests()
    if not tests:                                           # first build: run the tests so the report has real results
        tests = qa_tests.run_all()
        db.save_tests(tests)
    passed = sum(t["passed"] for t in tests)
    metrics = _metrics()
    E = escape

    buf = io.BytesIO()
    doc = Doc(buf)
    s = []

    # ------------------------------------------------------------------ cover
    s += [Spacer(1, 3.2 * cm), P("GROUP PRACTICAL ASSIGNMENT", "sub"), Spacer(1, 6), P("AI Agents Suite", "title"), Spacer(1, 6),
          P("Three Python AI agents: an Academic Advisor, a Supermarket Support Agent and a Personal Finance Agent, with Gemini-assisted "
            "research and a shared database.", "sub"), Spacer(1, 1.2 * cm),
          table([["Institution", UNIVERSITY], ["Unit", UNIT], ["Assignment", "Group practical assignment, 100 marks"],
                 ["Questions implemented", "Question 1 (Academic Advisory), Question 3 (Customer Support), Question 5 (Personal Finance)"],
                 ["Report date", dt.date.today().strftime("%d %B %Y")], ["Group size", f"{len(members)} of 6 members recorded"]],
                [4.2 * cm, 12.6 * cm], header=False),
          Spacer(1, 0.8 * cm), P("<b>Group members</b>", "body")]
    if members:
        s.append(table([["#", "Name", "Registration no."]] + [[i, E(m["name"]), E(m["reg_no"] or "-")] for i, m in enumerate(members, 1)],
                       [1 * cm, 9 * cm, 6.8 * cm]))
    else:
        s.append(P("No members have been added yet. Open the Group and report page of the app and add the members, then generate this report again."))
    s.append(PageBreak())

    s += [Paragraph("Contents", ParagraphStyle("Hc", parent=ST["H1"]))]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName="Helvetica", fontSize=10.5, leading=19, textColor=colors.HexColor(INK))]
    s += [toc, PageBreak()]

    # ------------------------------------------------------------------ 1 introduction
    s += [P("1. Introduction and background", "H1"),
          P("Artificial intelligence lets software perceive information, analyse a situation, decide what to do and act on the decision. "
            "This project builds three practical intelligent agents in Python and presents them in one web application "
            "(Streamlit), so each can be demonstrated live. The three questions selected from the handout are:"),
          *bullets(["<b>Question 1 - Student Academic Advisory Agent.</b> Monitors marks, predicts final results and coaches the student.",
                    "<b>Question 3 - Customer Support Agent.</b> We chose a <b>supermarket</b>. It understands questions, answers from a knowledge base "
                    "and a live product catalogue, and researches questions it has no stored answer for.",
                    "<b>Question 5 - Personal Finance Agent.</b> Categorises spending, finds patterns, forecasts and recommends savings."]),
          P("Beyond the handout minimum, the suite adds a shared SQLite database (memory), a Gemini API integration with a keyless fallback, "
            "a command-line interface, downloadable source code and datasets, an automated test runner, and this self-generating report. "
            "Following the handout's guidance on AI APIs, Gemini <i>assists</i> the agents; the core intelligence (regression, intent "
            "classification, retrieval, rule-based reasoning, pattern detection) is implemented and explained by the group, and every agent "
            "still works when no API key is available."),
          P("2. Problem statements", "H1"),
          table([["Agent", "Real-world problem", "Who is affected"],
                 ["Academic Advisor", "Students learn that they are failing a unit only when results are released, which is too late to act. "
                  "Study resources are also generic and do not match the unit being studied.", "University students and academic advisors"],
                 ["Supermarket Support", "Customers repeatedly ask about prices, stock, offers, delivery and returns. Urgent or food-safety complaints "
                  "wait in the same queue as simple questions, and a closed FAQ cannot answer anything new.", "Shoppers and store staff"],
                 ["Finance Agent", "People do not know where their money goes until the month has been overspent, and manual categorising is tedious.",
                  "Individuals and young professionals"]],
                [3.1 * cm, 10.2 * cm, 3.5 * cm]),
          P("3. Objectives", "H1"),
          table([["Agent", "Objectives"],
                 ["Academic Advisor", "Accept student details and marks; identify weak units; predict unfinished exams; recommend improvements, study "
                  "activities and a weekly plan; research resources for <b>any</b> unit name; answer questions in a chat; produce a personal report; "
                  "act autonomously (study calendar, lecturer emails)."],
                 ["Supermarket Support", "Understand natural-language questions; identify intent; retrieve answers; look up live product price, offer, "
                  "stock and aisle; handle unknown questions by researching them with cited sources; keep the conversation; learn from feedback; triage "
                  "an inbox, raise tickets and escalate safety issues."],
                 ["Finance Agent", "Record income and spending; categorise automatically and learn from corrections; find patterns and waste; forecast "
                  "spending; recommend savings; plan a savings goal; run an autopilot (budgets, alerts, transfers, dispute drafts)."]],
                [3.1 * cm, 13.7 * cm]),
          PageBreak()]

    # ------------------------------------------------------------------ 4 agent and environment
    s += [P("4. The intelligent agents and their environment", "H1"),
          P("Each agent follows the same cycle: <b>perceive, analyse, decide, recommend, act</b>. The cycle is displayed in the app for each agent "
            "(Agent actions, Inbox triage and Autopilot tabs) and for all three at once on the Overview page."),
          table([["", "Academic Advisor", "Supermarket Support", "Finance Agent"],
                 ["Environment", "A student's marks, attendance and study habits", "A stream of customer messages, the product catalogue, the web",
                  "Income and a stream of transactions"],
                 ["Percepts", "CAT, assignment and exam marks; attendance; study hours", "Free text; entities (order numbers, amounts); urgency, "
                  "safety and frustration cues", "Descriptions, amounts, dates"],
                 ["Actuators", "Study calendar (.ics), lecturer emails, report", "Replies, tickets, routing, learned answers", "Budgets, alerts, transfer schedule, dispute drafts"],
                 ["Decides", "Who needs intervention; risk level; hours per unit", "Answer, show product, clarify, research, or escalate",
                  "Budget limits, saving amount, which alerts to raise"],
                 ["Performance measure", "Prediction error on unseen records", "Intent accuracy on unseen phrasings; satisfaction", "Categoriser accuracy; forecast back-test"],
                 ["Agent type", "Model-based, utility-guided", "Goal-based with learning", "Learning agent"]],
                [2.6 * cm, 4.6 * cm, 4.9 * cm, 4.7 * cm]),
          Spacer(1, 6),
          P("5. System design and architecture", "H1"),
          P("The application is layered. Pages (interface) call the three agent engines, which are plain Python modules that can be tested "
            "without the interface. The engines use shared services: the researcher, the Gemini client and the database. The same engines are used "
            "by the command-line interface and by the test suite."),
          _img(architecture_png(), 16.8 * cm, 10.5 * cm),
          P("Figure 1: Layered architecture.", "small"), Spacer(1, 6),
          P("Database design", "H2"),
          P("One SQLite file holds everything the system must remember. The 6-member limit is enforced by a <b>database trigger</b> as well as by "
            "the form, so it cannot be bypassed. The passphrase is stored only as a salted PBKDF2-SHA256 hash (200,000 rounds), with a lock-out after "
            "5 wrong attempts. Editing and deleting always need the passphrase."),
          table([["Table", "Remembers"],
                 ["members, member_audit", "Names, registration numbers, roles, responsibilities (maximum 6) and a history of changes"],
                 ["settings", "Passphrase hash and salt, lock-out state, who may add members"],
                 ["support_examples, support_penalty, support_qa, support_topics, support_tickets",
                  "Phrases the support agent learned, topic penalties from negative feedback, researched answers a customer approved, "
                  "topics added by staff, tickets"],
                 ["fin_corrections", "Spending categories the user corrected"],
                 ["research_cache", "Researched units and questions, so the same lookup is not repeated"],
                 ["products", "Product catalogue: price, offer, stock, aisle"],
                 ["test_runs", "Latest automated test results (shown in section 9)"]],
                [6.3 * cm, 10.5 * cm])]

    # ------------------------------------------------------------------ 6 techniques
    s += [P("6. AI techniques and algorithms implemented", "H1"),
          P("Academic Advisor", "H2"),
          *bullets(["<b>Ridge regression (scikit-learn)</b> predicts the final mark of units whose exam is not yet written, from coursework average, trend, "
                    "attendance and study hours. It also produces a likely range. Final mark = 30% coursework + 70% exam (an assumption that can be changed).",
                    "<b>Expert-system rules</b> diagnose each unit (declining trend, inconsistent marks, low attendance, borderline), assign a risk level "
                    "and choose study activities.",
                    "<b>Planning:</b> study hours are shared across units in proportion to a priority score; a goal solver computes the exam mark needed for a target grade.",
                    "<b>Research-based resource recommendation:</b> for any unit name the agent researches resources (Gemini with Google Search, then "
                    "Wikipedia and Open Library, then offline matching). The offline layer matches the unit to a subject area with TF-IDF cosine similarity and "
                    "builds search links from the unit name, so an unknown unit is never given a random list.",
                    "<b>Conversational interface</b> with intent rules, what-if and goal calculations, and a research fallback for questions no rule covers."]),
          P("Supermarket Support Agent", "H2"),
          *bullets(["<b>Intent classification:</b> TF-IDF word and character n-grams with Logistic Regression over 23 topics.",
                    "<b>Information retrieval:</b> TF-IDF cosine similarity over the knowledge base. The two scores are blended: "
                    "0.6 x classifier probability + 0.4 x min(1, 2 x similarity) - feedback penalty.",
                    "<b>Decision policy:</b> confident answer; product lookup in the live catalogue; \"did you mean\" when unsure; "
                    "research when nothing in the knowledge base fits; escalation to staff; honest \"I do not know\" when research finds nothing.",
                    "<b>Not a closed system:</b> unknown questions are researched (Gemini with Google Search, or Wikipedia) and answered with cited sources. "
                    "Food-safety words always go to a person, never to research.",
                    "<b>Learning:</b> thumbs-up adds the phrasing as a training example, or stores an approved researched answer; thumbs-down penalises a topic; "
                    "corrections add labelled examples. The classifier retrains instantly and everything is saved in the database.",
                    "<b>Context:</b> short follow-ups are merged with the previous question. <b>Autonomy:</b> inbox triage with routing, priority and tickets."]),
          P("Finance Agent", "H2"),
          *bullets(["<b>Expense categoriser:</b> TF-IDF + Logistic Regression trained on Kenyan merchant descriptions; learns from corrections (stored in the "
                    "database). When confidence is low, Gemini may choose a category, restricted to the agent's own category list.",
                    "<b>Pattern recognition:</b> weekday profile, recurring bills, small-purchase leakage, duplicates, and robust (median absolute deviation) outliers.",
                    "<b>Prediction:</b> Ridge regression and per-category exponentially weighted forecasts, chosen by a back-test against a moving average.",
                    "<b>Recommendation:</b> 50/30/20 guideline excess detection and ranked, quantified saving strategies; a goal planner; an autopilot that sets budgets, "
                    "schedules transfers, raises alerts and drafts dispute messages."]),
          P("Role of Gemini", "H2"),
          P("Gemini is used for three things: researching units with Google Search grounding, answering questions that have no stored keywords, and "
            "helping the finance categoriser and plan. The API key is entered in the sidebar for the session, or set as a server secret; it is never written to "
            "the database or to a file. Every Gemini call returns an explicit success flag, and when it fails the agent falls back to the keyless method, so the "
            "agents keep working. Gemini answers are labelled and carry their sources.")]

    # ------------------------------------------------------------------ 7 source code
    s += [P("7. Python source code", "H1"),
          P("The complete source is delivered with the application (download buttons on every agent page and on the Testing and data page, plus "
            "<font face='Courier'>python cli.py</font>). Libraries used: NumPy, Pandas, scikit-learn, Matplotlib, NetworkX, SQLite, Streamlit, Plotly, "
            "ReportLab and Requests. Source files:")]
    loc = _loc()
    s += [table([["File", "Lines", "Purpose"]] + [[f"<font face='Courier'>{f}</font>", n, d] for f, n, d in loc], [4.6 * cm, 1.6 * cm, 10.6 * cm]),
          P(f"Total: {sum(n for _, n, _ in loc):,} lines of Python.", "small"), Spacer(1, 6),
          P("Two short extracts show the design. First, the database trigger that enforces the six-member limit:", "body"),
          Preformatted("CREATE TRIGGER members_limit BEFORE INSERT ON members\nWHEN (SELECT COUNT(*) FROM members) >= 6\n"
                       "BEGIN SELECT RAISE(ABORT, 'member limit reached'); END;", ST["code"]),
          P("Second, the support agent's blended confidence score:", "body"),
          Preformatted("p, c = probs.get(i, 0.0), float(sim_by[i])        # classifier probability, retrieval similarity\n"
                       "s = 0.6 * p + 0.4 * min(1.0, c * 2.0) - self.penalty.get(i, 0.0)", ST["code"]),
          PageBreak()]

    # ------------------------------------------------------------------ 8 screenshots
    s += [P("8. Screenshots of the working applications", "H1")]
    shots = [("overview", "Overview and autonomous demo"), ("advisor", "Academic Advisor"), ("advisor_research", "Researching a unit that is not in the sample list"),
             ("support", "Supermarket Support Agent"), ("support_research", "Support Agent: researched answer with sources"), ("finance", "Finance Agent"),
             ("group", "Group members page"), ("testing", "Testing page")]
    shot_dir = ROOT / "assets" / "screenshots"
    n_shots = 0
    for key, cap in shots:
        f = shot_dir / f"{key}.png"
        if f.exists():
            s.append(KeepTogether([_img(f, 16.6 * cm, 9.6 * cm), P(f"Figure: {cap}.", "small"), Spacer(1, 6)]))
            n_shots += 1
    if not n_shots:
        s.append(P("Screenshots are inserted automatically when the <font face='Courier'>assets/screenshots</font> folder contains images."))
    s.append(PageBreak())

    # ------------------------------------------------------------------ 9 testing
    s += [P("9. Testing procedures and results", "H1"),
          P(f"The automated suite (<font face='Courier'>qa_tests.py</font>, also on the Testing page) ran <b>{len(tests)} tests: {passed} passed, "
            f"{len(tests) - passed} failed</b> at the time this report was built. Each test uses a throw-away database. Tests that would need the internet or a "
            "Gemini key use a mock server reply and say so in their description."),
          table([["Area", "Test", "Result", "What was checked"]] + [[E(t["agent"]), E(t["name"]), "Pass" if t["passed"] else "<b>FAIL</b>", E(t["detail"])] for t in tests],
                [2.6 * cm, 4.4 * cm, 1.6 * cm, 8.2 * cm]),
          P("Measured model results", "H2"),
          table([["Agent", "Measure", "Result"]] + metrics, [3.4 * cm, 8.4 * cm, 5 * cm]),
          P("Honest limitations of the evaluation. The advisor and finance data are synthetic, so their figures show that the method works and are not "
            "a claim of real-world accuracy. The support test set has 20 phrasings, so one error moves the score by five points; the phrasings were written "
            "after the training data and the agent was not tuned to them. Live Gemini, Wikipedia and Open Library calls cannot be reached from the "
            "development sandbox, so those paths are verified with mock responses and must be demonstrated live with an API key and internet access.", "small"),
          PageBreak()]

    # ------------------------------------------------------------------ 10 challenges
    s += [P("10. Challenges encountered", "H1"),
          *bullets(["<b>A closed knowledge base.</b> The first versions only answered what was stored and gave generic study resources. We added a research "
                    "layer (Gemini, Wikipedia, offline matching) and made the agents say where an answer came from.",
                    "<b>Wrong matches from fuzzy text.</b> \"System programming\" was matched to \"Database Systems\" because of the shared word. We "
                    "ignored generic words when matching unit names.",
                    "<b>Product questions versus other topics.</b> \"How long can I keep cooked rice?\" mentions a product but is not a price question. "
                    "The catalogue is used only when the message looks like a product question.",
                    "<b>Safety.</b> Messages about feeling sick or expired food must never be answered by a researched guess, so they are escalated first.",
                    "<b>Who may edit the group list.</b> The brief says members add their own details yet only one person may edit, delete or add. We made "
                    "editing and deleting always need the passphrase, and added a setting for whether adding is open or passphrase-only.",
                    "<b>Persistence on free hosting.</b> SQLite files on some hosts are erased on restart. We added JSON backup and restore and the DB_PATH setting.",
                    "<b>Light and dark themes.</b> Canvas tables ignore the theme, so tables, cards and charts are drawn with a custom flat design system.",
                    "<b>Small test sets and synthetic data</b> limit what the accuracy figures can claim."]),
          P("11. Conclusion and recommendations", "H1"),
          P("The project delivers three working AI agents that perceive, analyse, decide, recommend and act, backed by a database that gives them memory "
            "and by a research layer that stops them being limited to stored answers. The agents are covered by automated tests and can be run from a "
            "web interface or the command line."),
          P("Recommendations", "H2"),
          *bullets(["Replace the synthetic advisor and finance data with real, consented data before drawing real conclusions.",
                    "Replace the sample store policies and product catalogue with the real store's, and connect tickets to the store's helpdesk or email.",
                    "Set GEMINI_API_KEY and GROUP_PASSPHRASE as server secrets, and host the database on a persistent disk or a managed database.",
                    "Add more held-out test phrasings and measure Gemini-assisted answers against a human-checked set.",
                    "Add authentication per user if the system is used beyond a class demonstration."]),
          PageBreak()]

    # ------------------------------------------------------------------ group participation etc.
    s += [P("12. Group participation", "H1"),
          P("The handout lists a suggested division of responsibilities for four members. Our group has six members, so two extra roles were added. "
            "The table below is read from the group database, where each member entered their own details.")]
    if members:
        s.append(table([["#", "Member", "Reg. no.", "Role", "Responsibilities", "Agents"]] +
                       [[i, E(m["name"]), E(m["reg_no"] or "-"), E(m["role"] or "-"), E(m["responsibilities"]), E(m["agents"] or "All")]
                        for i, m in enumerate(members, 1)], [0.7 * cm, 3 * cm, 2.3 * cm, 3.2 * cm, 5.6 * cm, 2 * cm]))
    else:
        s.append(P("<b>No members are recorded yet.</b> Each member should open the Group and report page and add their name and responsibilities."))
    s += [P("Suggested division from the handout, for reference:", "small"),
          table([["Member", "Responsibility"], ["Member 1", "Problem analysis, requirements and system design"], ["Member 2", "Python programming and core functionality"],
                 ["Member 3", "AI algorithms, intelligence and decision-making"], ["Member 4", "User interface, testing and documentation"],
                 ["Members 5 and 6 (added)", "Database, deployment and security; research integration (Gemini API) and data"]], [4.2 * cm, 12.6 * cm]),
          P("All members should understand the complete system and be prepared to explain the implementation.", "small"),
          P("13. Mapping to the assessment criteria", "H1"),
          table([["Assessment component", "Marks", "Where it is evidenced"],
                 ["Problem identification and objectives", "10", "Sections 2 and 3"],
                 ["AI concepts and techniques", "20", "Sections 4 and 6; the Agent action tabs in the app"],
                 ["Python implementation", "25", "Section 7; the source-code downloads; the command-line interface"],
                 ["Intelligent decision-making / agent behaviour", "15", "Perceive-analyse-decide-recommend-act panels; inbox triage; finance autopilot; research decisions"],
                 ["User interface and additional functionality", "10", "Light/dark/system theme; database; downloads; Gemini; PDF report; command line"],
                 ["Testing and evaluation", "10", "Section 9; the Testing page"],
                 ["Group presentation and demonstration", "10", "Live demonstration using the Overview page demo and each agent page"]],
                [6.2 * cm, 1.6 * cm, 9 * cm]),
          P("14. Deliverables checklist", "H1"),
          table([["Handout deliverable", "Where"],
                 ["Three or more completed Python AI applications", "Academic Advisor, Support Agent, Finance Agent"],
                 ["Python source code for each application", "Download button on each agent page; Testing and data page"],
                 ["Project report", "This PDF, generated from the Group and report page"],
                 ["Screenshots", "Section 8"],
                 ["Sample datasets and testing results", "Testing and data page; section 9"],
                 ["Live demonstration", "Run the app and follow the demo script in README.md"]], [7 * cm, 9.8 * cm]),
          Spacer(1, 10),
          P("Appendix: how to run", "H2"),
          Preformatted("pip install -r requirements.txt\nstreamlit run app.py\npython qa_tests.py        # automated tests\n"
                       "python cli.py support \"do you deliver?\"\n# optional: GEMINI_API_KEY, GROUP_PASSPHRASE, DB_PATH", ST["code"])]

    doc.multiBuild(s)
    return buf.getvalue()


def _metrics() -> list[list]:
    """Fresh model measurements for the report tables."""
    import advisor_engine as adv
    import finance_engine as fin
    from db import Database
    from support_engine import SupportBot
    import os, tempfile
    d = Database(os.path.join(tempfile.mkdtemp(), "m.db"))
    mr = adv.model_report()
    bot = SupportBot(d)
    h, cv = bot.evaluate_heldout(), bot.evaluate()
    cat = fin.Categoriser()
    fh = fin.categoriser_heldout(cat)
    df, _ = fin.sample_data(cat)
    f = fin.forecast(df)
    return [["Academic Advisor", "Ridge regression: R-squared and mean error on 1,000 unseen synthetic records", f"{mr['r2']:.2f}; {mr['mae']:.1f} marks"],
            ["Support Agent", f"Intent accuracy on {h['n']} unseen phrasings", f"{h['accuracy']:.0%}"],
            ["Support Agent", "5-fold cross-validation on training phrases", f"{cv['accuracy']:.0%}" if cv["accuracy"] else "-"],
            ["Finance Agent", f"Categoriser accuracy on {fh['n']} unseen descriptions", f"{fh['accuracy']:.0%}"],
            ["Finance Agent", "Forecast back-test, error per day on the last 7 days",
             f"{f['mae_model']:,.0f} vs {f['mae_base']:,.0f} baseline" if f["mae_model"] is not None else "-"]]
