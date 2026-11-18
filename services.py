"""Shared services for the Streamlit pages: the database, Gemini and the researcher, plus the sidebar control panel."""
from __future__ import annotations

import streamlit as st

from db import Database
from llm import Gemini
from research import Researcher

ss = st.session_state


@st.cache_resource
def database() -> Database:
    return Database()


def gemini() -> Gemini:
    return Gemini(key=ss.get("gemini_key"), db=database())


def researcher() -> Researcher:
    return Researcher(database(), gemini(), use_web=ss.get("use_web", True))


def intelligence_label() -> str:
    """Short text for the research mode that is active right now."""
    g = gemini()
    if g.available():
        return f"Gemini ({g.model}) with Google Search"
    return "Wikipedia and Open Library" if ss.get("use_web", True) else "Offline"


def _save_gemini():
    """Callback for the Save button: the group passphrase is needed to change what is stored for everyone."""
    db = database()
    key, model = ss.get("gemini_key", ""), ss.get("gem_model_pick") or ss.get("gem_model_text", "")
    if db.passphrase_is_set():
        ok, msg = db.verify_passphrase(ss.get("gem_pass", ""))
        if not ok:
            ss.gem_msg = ("error", msg)
            return
    if not (key or "").strip() and not (model or "").strip():
        ss.gem_msg = ("error", "Enter a key or a model to save.")
        return
    db.set_gemini(key, model)
    ss.gemini_key = ""          # the saved copy is used from now on
    ss.gem_msg = ("ok", "Saved in the database. All agents will use it on every visit.")


def _remove_gemini():
    db = database()
    if db.passphrase_is_set():
        ok, msg = db.verify_passphrase(ss.get("gem_pass", ""))
        if not ok:
            ss.gem_msg = ("error", msg)
            return
    db.clear_gemini_key()
    ss.gem_msg = ("ok", "Saved key removed.")


def sidebar_panel():
    db = database()
    with st.sidebar:
        with st.expander("Gemini and research", icon=":material/psychology:"):
            g = gemini()
            cfg = db.gemini_settings()
            st.caption(f"Model: **{g.model}**  \nKey: " + (f"{g.key_source()}" + (f" ({cfg['masked']})" if cfg["has_key"] and g.key_source() == "saved in the database" else "")
                                                         if g.available() else "none"))
            st.text_input("Gemini API key", type="password", key="gemini_key", placeholder="paste a new key",
                          help="Used straight away for this session. Press Save to keep it in the database for every visit.")
            models = ss.get("gem_models") or []
            if models:
                st.selectbox("Model", models, index=models.index(g.model) if g.model in models else 0, key="gem_model_pick")
            else:
                st.text_input("Model", value=g.model, key="gem_model_text", help="For example gemini-3.8-flash")
            if st.button("Load models from Google", icon=":material/refresh:", key="gem_load", disabled=not g.available()):
                names, err = g.list_models()
                if names:
                    ss.gem_models = names
                    st.rerun()
                else:
                    st.error(err)
            if db.passphrase_is_set():
                st.text_input("Group passphrase (needed to save)", type="password", key="gem_pass")
            else:
                st.caption("No group passphrase is set yet, so anyone can save here. Set one on the Group page.")
            c1, c2 = st.columns(2)
            c1.button("Save", icon=":material/save:", key="gem_save", on_click=_save_gemini, type="primary")
            c2.button("Remove key", icon=":material/delete:", key="gem_remove", on_click=_remove_gemini, disabled=not cfg["has_key"])
            if ss.get("gem_msg"):
                (st.success if ss.gem_msg[0] == "ok" else st.error)(ss.gem_msg[1])
            if g.available() and st.button("Test the key", icon=":material/bolt:", key="gem_test"):
                ok, msg = g.test()
                (st.success if ok else st.error)("Gemini replied." if ok else msg)
            st.toggle("Allow live web research", value=True, key="use_web",
                      help="Wikipedia and Open Library lookups when there is no Gemini key.")
            st.caption("Active: " + intelligence_label() + ". The saved key is stored in the database file, so keep that file private; "
                       "visitors use it but cannot read it.")


@st.cache_data(show_spinner=False)
def _zip(agent: str) -> bytes:
    import sourcezip
    return sourcezip.agent_zip(agent)


def source_button(agent: str):
    """Download-source-code button for one agent."""
    import sourcezip
    a = sourcezip.AGENTS[agent]
    st.download_button("Download this agent's source code", _zip(agent), file_name=f"{agent}_agent_source.zip",
                       mime="application/zip", icon=":material/folder_zip:", key=f"src_btn_{agent}")
