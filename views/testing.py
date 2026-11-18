"""Testing and data: live test runner, sample datasets, source-code downloads."""
import pandas as pd
import streamlit as st

import datasets
import qa_tests
import services
import sourcezip
import theme

ss = st.session_state
db = services.database()

theme.hero("Testing and data",
           "Run the automated tests, look at the results that go into the report, and download the sample datasets and source code.")

t_test, t_data, t_src = st.tabs([":material/fact_check: Tests", ":material/dataset: Sample datasets", ":material/code: Source code"])

with t_test:
    st.caption("Every test runs on a throw-away database. Tests that would need the internet or a Gemini key use a mock server "
               "reply (their names say so), because they must not depend on a service being reachable.")
    if st.button("Run all tests", icon=":material/play_arrow:", type="primary", key="qa_run"):
        with st.spinner("Running tests..."):
            rows = qa_tests.run_all()
            db.save_tests(rows)
    rows = db.load_tests()
    if rows:
        passed = sum(r["passed"] for r in rows)
        theme.cards([("Tests passed", f"{passed} of {len(rows)}", "latest run", "good" if passed == len(rows) else "bad"),
                     ("Agents covered", str(len({r["agent"] for r in rows})), "plus database and research"),
                     ("Total time", f"{sum(r['seconds'] for r in rows):.1f} s", "")])
        st.write("")
        theme.html_table(pd.DataFrame([{"Area": r["agent"], "Test": r["name"], "Result": "Pass" if r["passed"] else "Fail",
                                        "What was checked": r["detail"]} for r in rows]),
                         tone_col="Result", tones={"Pass": "good", "Fail": "bad"})
        st.caption("Accuracy figures come from small test sets and synthetic data. They show the method works; they are not a "
                   "promise of real-world accuracy.")
    else:
        theme.callout("No test run is stored yet. Press Run all tests.", "info")

with t_data:
    st.caption("The data the agents are demonstrated with. All of it is sample or generated data.")
    items = [("Student marks (Academic Advisor)", "sample_marks.csv", datasets.advisor_marks_csv),
             ("Customer messages (Support Agent inbox)", "sample_inbox.txt", datasets.support_inbox_txt),
             ("Unseen test phrasings (Support Agent)", "support_heldout.csv", datasets.support_heldout_csv),
             ("Product catalogue (Support Agent)", "products.csv", lambda: datasets.products_csv(db)),
             ("Transactions (Finance Agent)", "sample_transactions.csv", datasets.finance_transactions_csv)]
    for label, fn, make in items:
        c1, c2 = st.columns([4, 2])
        c1.markdown(f"**{label}**  \n`{fn}`")
        c2.download_button("Download", make(), file_name=fn, icon=":material/download:", key=f"ds_{fn}")

with t_src:
    st.caption("Each agent can be downloaded on its own and runs with `streamlit run app.py`. The suite download contains everything.")
    for key, a in sourcezip.AGENTS.items():
        c1, c2 = st.columns([4, 2])
        c1.markdown(f"**{a['title']}**  \n{a['blurb']}")
        c2.download_button("Download source code", sourcezip.agent_zip(key), file_name=f"{key}_agent_source.zip",
                           mime="application/zip", icon=":material/folder_zip:", key=f"src_{key}")
    c1, c2 = st.columns([4, 2])
    c1.markdown("**Whole suite**  \nAll three agents, the database, report generator, tests and README.")
    c2.download_button("Download the suite", sourcezip.suite_zip(), file_name="ai_agents_suite.zip", mime="application/zip",
                       icon=":material/folder_zip:", key="src_suite", type="primary")
