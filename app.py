"""
AI Agents Suite - one Streamlit app that hosts the three agents.

Run with:   streamlit run app.py
"""
import streamlit as st

import theme

st.set_page_config(page_title="AI Agents Suite", layout="wide")

with st.sidebar:
    theme.theme_switch()
theme.apply()

pages = [
    st.Page("views/advisor.py", title="Academic Advisor", icon=":material/school:", default=True, url_path="advisor"),
    st.Page("views/support.py", title="Support Agent", icon=":material/support_agent:", url_path="support"),
    st.Page("views/finance.py", title="Finance Agent", icon=":material/account_balance_wallet:", url_path="finance"),
]
st.navigation(pages).run()
