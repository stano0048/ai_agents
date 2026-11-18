"""Builds the source-code downloads: one zip per agent (runs on its own) and one for the whole suite."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMON = ["theme.py", "services.py", "db.py", "llm.py", "research.py", "sourcezip.py"]
EXTRA = ["requirements.txt", ".streamlit/config.toml"]

AGENTS = {
    "advisor": dict(title="Academic Advisor", files=["advisor_engine.py", "advisor_chat.py", "views/advisor.py"], icon=":material/school:",
                    blurb="Ridge regression prediction, expert-system rules, unit research (Gemini / Wikipedia), conversational advisor."),
    "support": dict(title="Supermarket Support Agent", files=["support_engine.py", "views/support.py"], icon=":material/support_agent:",
                    blurb="TF-IDF + Logistic Regression intents, retrieval, product catalogue, live research, learning from feedback."),
    "finance": dict(title="Finance Agent", files=["finance_engine.py", "views/finance.py"], icon=":material/account_balance_wallet:",
                    blurb="Expense categoriser, pattern detection, forecasting, savings strategies, autopilot."),
}


def standalone_app(agent: str) -> str:
    a = AGENTS[agent]
    page = a["files"][1]
    return f'''"""{a["title"]} - runs on its own.   streamlit run app.py"""
import streamlit as st

import services
import theme

st.set_page_config(page_title="{a["title"]}", layout="wide")
with st.sidebar:
    theme.theme_switch()
    services.sidebar_panel()
theme.apply()
st.navigation([st.Page("{page}", title="{a["title"]}", icon="{a["icon"]}", default=True)]).run()
'''


def _readme(agent: str) -> str:
    a = AGENTS[agent]
    return f"""# {a["title"]}

{a["blurb"]}

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

Optional: set `GEMINI_API_KEY` (or paste a key in the sidebar) so the agent can research with Gemini and Google Search.
Without a key it falls back to Wikipedia / Open Library and offline matching.
Memory (learning, corrections, research cache) is kept in `data/agents.db` (override with `DB_PATH`).

This agent is one part of the AI Agents Suite; the full project README explains the design, testing and deployment.
"""


def _add(z: zipfile.ZipFile, rel: str, prefix: str = ""):
    p = ROOT / rel
    if p.exists():
        z.write(p, prefix + rel)


def agent_zip(agent: str) -> bytes:
    a = AGENTS[agent]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in COMMON + a["files"] + EXTRA:
            _add(z, rel)
        z.writestr("app.py", standalone_app(agent))
        z.writestr("README.md", _readme(agent))
        z.writestr("data/.gitkeep", "")
    return buf.getvalue()


def suite_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if (p.is_dir() or "__pycache__" in rel.parts or rel.parts[0] == "data" and p.suffix in (".db", ".json")
                    or p.suffix in (".zip", ".pyc", ".log")):
                continue
            z.write(p, str(rel))
        z.writestr("data/.gitkeep", "")
    return buf.getvalue()
