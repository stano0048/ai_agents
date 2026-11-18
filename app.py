"""
AI Agents Suite - one Streamlit app that hosts the three agents.

Run with:   streamlit run app.py
"""
import streamlit as st

import services
import theme

st.set_page_config(page_title="AI Agents Suite", layout="wide")

with st.sidebar:
    theme.theme_switch()
    services.sidebar_panel()
theme.apply()

pages = [
    st.Page("views/home.py", title="Overview", icon=":material/hub:", default=True, url_path="overview"),
    st.Page("views/advisor.py", title="Academic Advisor", icon=":material/school:", url_path="advisor"),
    st.Page("views/support.py", title="Support Agent", icon=":material/support_agent:", url_path="support"),
    st.Page("views/finance.py", title="Finance Agent", icon=":material/account_balance_wallet:", url_path="finance"),
    st.Page("views/group.py", title="Group and report", icon=":material/groups:", url_path="group"),
    st.Page("views/testing.py", title="Testing and data", icon=":material/fact_check:", url_path="testing"),
]
st.navigation(pages).run()
