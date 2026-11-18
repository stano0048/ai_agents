# AI Agents Suite

Three Python AI agents in one Streamlit app, with a shared database, Gemini-assisted research, a group page and an
auto-generated PDF report. Flat colours (no gradients), no emojis, Material icons, and a System / Light / Dark theme control.

| Agent | Handout question | What it does |
|---|---|---|
| Academic Advisor | Q1 | Predicts final marks, finds weak units, recommends activities and **researches learning resources for any unit name** |
| Supermarket Support Agent | Q3 | Understands customer questions, answers from a knowledge base and the **live product catalogue**, **researches what it does not know**, learns from feedback |
| Finance Agent | Q5 | Categorises spending, finds patterns and waste, forecasts, recommends savings, runs an autopilot |

## Contents
1. [Quick start](#quick-start)
2. [How the agents work](#how-the-agents-work)
3. [Research and Gemini](#research-and-gemini)
4. [The database (memory)](#the-database-memory)
5. [Group members and the passphrase](#group-members-and-the-passphrase)
6. [The PDF report](#the-pdf-report)
7. [Source-code downloads, datasets and the command line](#source-code-downloads-datasets-and-the-command-line)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Demo script](#demo-script)
11. [Project layout](#project-layout)
12. [Honest limitations](#honest-limitations)
13. [How this meets the handout](#how-this-meets-the-handout)

## Quick start
```
pip install -r requirements.txt
streamlit run app.py
python qa_tests.py            # automated tests (no internet or key needed)
```
Optional environment variables:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini key (optional; the key can also be saved from the sidebar) |
| `GEMINI_MODEL` | Default model name (default `gemini-3.8-flash`; the sidebar setting overrides it) |
| `GROUP_PASSPHRASE` | Sets the group passphrase from the server, so nobody can claim it first |
| `DB_PATH` | Location of the SQLite file (default `data/agents.db`) |

**Gemini key and model, from the sidebar (Gemini and research):** paste the key, choose or type the model (press *Load models from Google* to list what your key can use), then **Save**. Both are stored in the database, so every agent uses them on every visit. Saving or removing needs the group passphrase once one is set. Typing a key without saving uses it for that browser session only. Note: the saved key sits in the database file in plain text and visitors use it (they cannot read it), so keep `data/agents.db` private and prefer the `GEMINI_API_KEY` server secret when you deploy publicly.

## How the agents work
Every agent follows one cycle: **perceive, analyse, decide, recommend, act**. The cycle is shown on screen
(Agent actions, Inbox triage and Autopilot tabs, and for all three on **Overview > Run all three agents**).

| | Academic Advisor | Supermarket Support | Finance Agent |
|---|---|---|---|
| Perceive | marks, attendance, study habits | free text, entities, urgency, safety words, mood | income, raw spending descriptions |
| Analyse | weak units, trends, risk | intent + retrieval + product match | categories, patterns, outliers |
| Decide | who needs intervention | answer, show product, clarify, research or escalate | budgets, saving amount, alerts |
| Solve | exam mark needed for a target | follow-ups from conversation memory | savings-goal planner |
| Recommend | activities, researched resources, hours | related topics, right department | ranked, quantified saving strategies |
| Act on its own | study calendar (.ics), lecturer emails | inbox triage, tickets, routing | budgets, transfer schedule, alerts, dispute drafts |

### Academic Advisor (Question 1)
- **Ridge regression** predicts unfinished exams from coursework average, trend, attendance and study hours (final mark = 30% coursework + 70% exam, an assumption you can change in `advisor_engine.py`).
- **Expert-system rules** diagnose each unit and pick study activities and a risk level.
- **Planner and goal solver**: study hours are shared by priority; "what do I need for a B?" is computed.
- **Conversational advisor** (`advisor_chat.py`) understands the *meaning* of a question, not keywords: Gemini reads it when a key is set, otherwise a local intent classifier (TF-IDF word + character n-grams, Logistic Regression, typo tolerant) decides. The unit is found separately, so a unit that is not in the marks list is researched by name, and a general or off-list question goes to the research layer. Numbers (what-if, target grade) are always calculated by the agent's own model.
- **Resources are researched from the unit name** (see below). Press *Research resources for all my units*, or use *Research any unit* (for example Entrepreneurship, System Programming).

### Supermarket Support Agent (Question 3)
- **NLP**: normalisation, synonyms, entity extraction (order numbers, amounts, dates, phone), urgency / food-safety / frustration cues.
- **Intent classifier**: TF-IDF word + character n-grams and Logistic Regression over 23 topics.
- **Retrieval**: TF-IDF cosine similarity over the knowledge base. Score = 0.6 x classifier probability + 0.4 x min(1, 2 x similarity) - feedback penalty.
- **Product catalogue**: price, offer, stock and aisle are read from the database `products` table, so product answers are never hard-coded.
- **Decision policy**: confident answer, catalogue answer, "did you mean", research, escalation, or an honest "I do not know".
- **Not closed**: when nothing stored fits, the agent researches the question and cites the sources. Food-safety messages ("I feel sick", "expired") skip research and go straight to a person with a high-priority ticket.
- **Learning**: a thumbs-up adds the phrasing as training data, or remembers an approved researched answer; thumbs-down lowers a topic's score; picking the right topic adds a corrected example. Everything is stored in the database, so it survives restarts.
- **Autonomy**: inbox triage answers, researches, routes, prioritises and opens tickets.
- The knowledge base and products are **sample content** for a fictional store. Replace them with the real store's.

### Finance Agent (Question 5)
- **Categoriser**: TF-IDF + Logistic Regression on Kenyan merchant descriptions. Corrections are stored in the database. When confidence is low, Gemini may pick a category, restricted to the agent's own list (at most 20 lookups per batch).
- **Patterns**: weekday profile, recurring bills, small-purchase leakage, duplicates, robust MAD outliers.
- **Forecast**: Ridge regression and per-category exponentially weighted forecasts, chosen by a back-test against a moving average.
- **Recommendations**: 50/30/20 guideline excess and ranked saving strategies; goal planner; autopilot. *Ask Gemini for a plain-language plan* rewrites the agent's own numbers (Gemini may not invent figures).

## Research and Gemini
`research.py` tries three layers and labels which one answered (`mode`):

1. **Gemini** (`llm.py`) with Google Search grounding when a key is set. The pages Gemini used are shown as sources.
2. **Live keyless**: Wikipedia summary and Open Library books, plus search links built from the topic.
3. **Offline**: the unit is matched to a subject area by TF-IDF cosine similarity and search links are built from the unit name. It never returns a random list, but it cannot read the web.

Gemini is used for: researching units, answering questions with no stored keywords, assisting the finance categoriser and plan. If a Gemini call fails (no key, quota, network) the agent silently falls back to the next layer. Researched results are cached in the database.
The key is stored only if you press Save in the sidebar (see Quick start); otherwise it stays in the session.

## The database (memory)
One SQLite file (`data/agents.db`). Tables:

| Table | Remembers |
|---|---|
| `members`, `member_audit` | up to 6 members and a change history |
| `settings` | passphrase hash and salt, lock-out state, who may add members, saved Gemini key and model |
| `support_examples`, `support_penalty`, `support_qa`, `support_topics`, `support_tickets` | what the support agent learned, topics staff added, tickets |
| `fin_corrections` | category corrections |
| `research_cache` | researched units and questions |
| `products` | the product catalogue |
| `test_runs` | latest test results (used in the report) |

## Group members and the passphrase
Open **Group and report**.

1. **The group leader sets the passphrase first**, before sharing the link (or set `GROUP_PASSPHRASE` on the server, which is safer: otherwise whoever opens the page first could claim it).
2. Send the app link to each member. Each adds their name, registration number, role, responsibilities and agents. **Only 6 entries are allowed.** The limit is enforced by a database trigger, not only by the form.
3. **Edit and delete always need the passphrase.** The passphrase is stored only as a salted PBKDF2-SHA256 hash (200,000 rounds); 5 wrong attempts lock the page for 60 seconds. "The system remembers" the passphrase in the database, so it is the same on every visit.
4. **Who may add**: the brief says members add their own entries yet "only one person can add". The setting under *Settings, backup and history* lets you choose: *anyone with the link* (default, members add themselves) or *only the passphrase holder*.
5. **Backup**: some hosts erase local files on restart. After everyone has added their details, download the JSON backup. You can restore it if the list is ever empty.

## The PDF report
**Group and report > Generate the PDF report** builds `group_report.pdf` from the live database and offers it for download. Anyone visiting the app can do this. It follows the handout's report outline (introduction, problems, objectives, agents and environment, design and architecture, techniques, source code, screenshots, testing, challenges, conclusion), plus group participation (members and responsibilities from the database), the assessment mapping and a deliverables checklist. Test results and model metrics are measured when the report is built. Screenshots come from `assets/screenshots/`.
Also available from the command line: `python cli.py report group_report.pdf`.

## Source-code downloads, datasets and the command line
- Every agent page has **Download this agent's source code** (a zip that runs on its own with `streamlit run app.py`). **Testing and data > Source code** has all agents and the whole suite.
- **Testing and data > Sample datasets**: student marks, customer messages, unseen test phrasings, product catalogue, transactions.
- Command line (the handout accepts a command-line interface):
```
python cli.py advisor --sample
python cli.py advisor --unit "Entrepreneurship"
python cli.py advisor --ask "recommend resources for system programming"
python cli.py support "do you deliver?"      # no text = interactive chat
python cli.py finance --sample
python cli.py members
python cli.py report group_report.pdf
```

## Testing
`python qa_tests.py` (or **Testing and data > Run all tests**) runs 38 tests on a throw-away database: database rules (6-member limit, duplicates, passphrase, lock-out, persistence, backup), research paths, advisor, support and finance behaviour. Tests that would need the internet or a Gemini key use **mock server replies** (their names say so).
Measured results are shown in the app and the report; they use small test sets and synthetic data (see limitations).

## Deployment
- **Streamlit Community Cloud**: push the project to GitHub, create the app with `app.py`, add secrets `GEMINI_API_KEY` and `GROUP_PASSPHRASE`. The local filesystem is **not guaranteed to persist**: download the members backup, or point `DB_PATH` at a persistent disk.
- **Any server / VM**: `streamlit run app.py --server.port 8501`; set `DB_PATH` to a persistent path.
- Share the app's URL with members so they can add their details.

## Demo script
1. **Overview**: show the status cards, press *Run all three agents* and read the perceive-analyse-decide-recommend-act panels.
2. **Academic Advisor**: *Load sample data*, Dashboard, Predictions, Agent actions (calendar and emails). Recommendations: *Research any unit*, type "Entrepreneurship", then "System Programming". Chat: "what do I need to get a B in Database Systems?"
3. **Support Agent**: chat "how much is cooking oil?", "do you have quinoa?", "how do I get a loyalty card?", then an open question ("how long can I keep cooked rice in the fridge?") to show research and sources. Give a thumbs-up and ask again to show it remembers. Inbox triage with the sample messages.
4. **Finance Agent**: *Load sample data*, Overview, Patterns, Forecast, Autopilot; correct a category and show it is remembered.
5. **Group and report**: members, passphrase, generate and open the PDF. **Testing and data**: Run all tests.

## Project layout
```
app.py                router, theme switch, sidebar
theme.py              flat light/dark design system
services.py           database, Gemini and researcher for the pages
db.py  llm.py  research.py
advisor_engine.py  support_engine.py  finance_engine.py     the AI logic (testable without the UI)
views/                home, advisor, support, finance, group, testing
report.py             PDF report      sourcezip.py  downloads      datasets.py  sample data
cli.py                command line    qa_tests.py   test suite     tests/test_agents.py  quick checks
assets/screenshots/   images used in the report
```

## Honest limitations
- **Live Gemini, Wikipedia and Open Library calls were not run during development** (the build environment had no access to them). Those paths are tested with mock responses. Please try them once with a real key and internet connection before the demonstration.
- The advisor's model is trained on **synthetic data**; the 30/70 weights and grade bands are assumptions. The finance sample data is generated.
- The support test set has 20 phrasings: one error moves the score by 5 points. Researched answers are general information, not the store's own records, and are labelled as such.
- Tickets and emails are drafted and stored, not sent anywhere.
- Passphrase protection is suited to a class project: sessions are not per-user accounts.

## How this meets the handout
| Handout requirement | Where |
|---|---|
| At least three AI applications (Q1, Q3, Q5) | The three agent pages |
| Real-world problems; perceive, analyse, decide, solve, recommend, act | Cycle panels; report sections 2-4 |
| Python, working interface (GUI / web / CLI) | Streamlit app and `cli.py` |
| Identifiable AI technique per project | Regression, rules, TF-IDF + Logistic Regression, retrieval, forecasting, research |
| Libraries: NumPy, Pandas, scikit-learn, Matplotlib, NetworkX, SQLite, Streamlit | All used (Matplotlib + NetworkX draw the report diagram) |
| AI APIs allowed, but concepts must be understood | Gemini assists; core models are our own and the report explains them |
| Report with sections 1-11 | Generated PDF |
| Screenshots, sample datasets, testing results, source code | PDF; Testing and data page; download buttons |
| Extras: database, dashboards, notifications, autonomous tasks | SQLite memory; dashboards; alerts and tickets; autopilot and triage |
| Groups of six (handout says four) | Group page, limit of six |
