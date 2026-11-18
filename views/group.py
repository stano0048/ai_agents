"""Group members, responsibilities and the downloadable group report."""
import json

import pandas as pd
import streamlit as st

import services
import theme
from db import MAX_MEMBERS, DuplicateMemberError, MemberLimitError

ss = st.session_state
db = services.database()

ROLES = ["Problem analysis, requirements and system design", "Python programming and core functionality",
         "AI algorithms, intelligence and decision-making", "User interface, testing and documentation",
         "Database, deployment and security", "Research integration (Gemini API) and data", "Other (describe below)"]
AGENT_NAMES = ["Academic Advisor", "Support Agent", "Finance Agent"]

theme.hero("Group and report",
           f"Each member adds their name and responsibilities here (up to {MAX_MEMBERS}). The details are stored in the database "
           "and appear in the group report, which anyone can download.")

ss.setdefault("group_unlocked", False)
members = db.list_members()
policy = db.add_policy()
unlocked = ss.group_unlocked

theme.cards([("Members", f"{len(members)} of {MAX_MEMBERS}", "entries used", "good" if len(members) == MAX_MEMBERS else None),
             ("Editing", "Unlocked" if unlocked else "Locked", "needs the group passphrase", "good" if unlocked else None),
             ("Who can add", "Anyone with the link" if policy == "open" else "Passphrase holder only", "edit and delete always need the passphrase")])
st.write("")

# ---------------------------------------------------------------- passphrase
st.subheader("Passphrase")
if not db.passphrase_is_set():
    theme.callout("No passphrase has been set yet. The first person to set it controls editing and deleting, so the group "
                  "leader should do this first, before sharing the link. (Better: set GROUP_PASSPHRASE on the server.)", "warn")
    with st.form("set_pass"):
        p1 = st.text_input("Choose a passphrase (at least 6 characters)", type="password")
        p2 = st.text_input("Repeat the passphrase", type="password")
        if st.form_submit_button("Save passphrase", icon=":material/key:", type="primary"):
            if p1 != p2:
                st.error("The two passphrases do not match.")
            else:
                ok, msg = db.set_passphrase(p1)
                if ok:
                    ss.group_unlocked = True
                    st.rerun()
                else:
                    st.error(msg)
elif not unlocked:
    st.caption("Enter the group passphrase to edit, delete" + (" or add" if policy != "open" else "") + " members.")
    with st.form("unlock"):
        att = st.text_input("Passphrase", type="password")
        if st.form_submit_button("Unlock", icon=":material/lock_open:", type="primary"):
            ok, msg = db.verify_passphrase(att)
            if ok:
                ss.group_unlocked = True
                st.rerun()
            else:
                st.error(msg)
else:
    if st.button("Lock editing", icon=":material/lock:", key="grp_lock"):
        ss.group_unlocked = False
        st.rerun()

# ------------------------------------------------------------------- members
st.subheader("Group members")
if members:
    theme.html_table(pd.DataFrame([{"#": i, "Name": m["name"], "Reg. no.": m["reg_no"] or "-", "Role": m["role"] or "-",
                                    "Responsibilities": m["responsibilities"], "Agents": m["agents"] or "All"}
                                   for i, m in enumerate(members, 1)]))
else:
    theme.callout("No members yet. Use the form below to add the first one.", "info")

can_add = len(members) < MAX_MEMBERS and (policy == "open" or unlocked) and db.passphrase_is_set()
if len(members) >= MAX_MEMBERS:
    theme.callout(f"The group is full ({MAX_MEMBERS} members). Unlock editing to change or remove someone.", "info")
elif not db.passphrase_is_set():
    st.caption("Set the passphrase first, then members can be added.")
elif not can_add:
    st.caption("Adding members is limited to the passphrase holder. Unlock editing above.")
else:
    st.markdown("**Add a member**")
    with st.form("add_member", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Full name")
        reg = c2.text_input("Registration number")
        role = st.selectbox("Main role", ROLES)
        resp = st.text_area("Responsibilities (what you did in building these agents)", height=100)
        ag = st.multiselect("Agents you worked on", AGENT_NAMES)
        if st.form_submit_button("Add member", icon=":material/person_add:", type="primary"):
            try:
                db.add_member(name, reg, "" if role.startswith("Other") else role, resp, ag)
                st.toast("Member added.", icon=":material/check:")
                st.rerun()
            except (ValueError, MemberLimitError, DuplicateMemberError) as e:
                st.error(str(e))

# ---------------------------------------------------------- edit and delete
if unlocked and members:
    st.markdown("**Edit or remove a member**")
    pick = st.selectbox("Member", members, format_func=lambda m: m["name"], key="grp_pick")
    with st.form("edit_member"):
        c1, c2 = st.columns(2)
        n = c1.text_input("Full name", value=pick["name"], key=f"e_n_{pick['id']}")
        r = c2.text_input("Registration number", value=pick["reg_no"], key=f"e_r_{pick['id']}")
        rl = st.text_input("Role", value=pick["role"], key=f"e_rl_{pick['id']}")
        rs = st.text_area("Responsibilities", value=pick["responsibilities"], height=100, key=f"e_rs_{pick['id']}")
        ags = st.multiselect("Agents", AGENT_NAMES, default=[a for a in pick["agents"].split(", ") if a in AGENT_NAMES],
                             key=f"e_a_{pick['id']}")
        if st.form_submit_button("Save changes", icon=":material/save:", type="primary"):
            try:
                db.update_member(pick["id"], n, r, rl, rs, ags)
                st.rerun()
            except (ValueError, DuplicateMemberError) as e:
                st.error(str(e))
    sure = st.checkbox(f"Yes, remove {pick['name']}", key=f"grp_sure_{pick['id']}")
    if st.button("Remove member", icon=":material/delete:", disabled=not sure, key="grp_del"):
        db.delete_member(pick["id"])
        st.rerun()

# -------------------------------------------------------------------- admin
if unlocked:
    with st.expander("Settings, backup and history", icon=":material/settings:"):
        pol = st.radio("Who may add members?", ["open", "passphrase"], index=0 if policy == "open" else 1, horizontal=True,
                       format_func=lambda x: "Anyone with the link" if x == "open" else "Only the passphrase holder",
                       help="Edit and delete always need the passphrase. Choose 'passphrase' to make adding passphrase-only too.")
        if pol != policy:
            db.set("add_policy", pol)
            st.rerun()
        st.markdown("**Change the passphrase**")
        with st.form("chg_pass"):
            o = st.text_input("Current passphrase", type="password")
            nw = st.text_input("New passphrase", type="password")
            if st.form_submit_button("Change passphrase", icon=":material/key:"):
                ok, msg = db.set_passphrase(nw, o)
                (st.success if ok else st.error)(msg)
        st.markdown("**Backup**")
        st.caption("Hosts such as Streamlit Community Cloud can erase local files when the app restarts. Download a backup after "
                   "everyone has added their details, and restore it if the list is ever empty.")
        st.download_button("Download members backup (JSON)", db.export_members(), file_name="group_members_backup.json",
                           mime="application/json", icon=":material/download:", key="grp_backup")
        up = st.file_uploader("Restore from a backup", type="json", key="grp_restore")
        if up is not None and st.button("Restore members", icon=":material/upload:", key="grp_restore_btn"):
            try:
                n = db.import_members(up.getvalue().decode("utf-8"))
                st.success(f"Restored {n} member(s). Duplicates and entries over the limit were skipped.")
            except (ValueError, UnicodeDecodeError, AttributeError):
                st.error("That file is not a valid backup.")
        aud = db.audit(8)
        if aud:
            import datetime as dt
            st.markdown("**Recent changes**")
            theme.html_table(pd.DataFrame([{"When": dt.datetime.fromtimestamp(a["ts"]).strftime("%d %b %H:%M"), "Action": a["action"],
                                            "Member": a["member"]} for a in aud]))

# -------------------------------------------------------------------- report
st.subheader("Group report")
st.caption("A PDF written by this system from the live database: members and responsibilities, the three agents, design, "
           "AI techniques, test results, challenges and conclusions. Anyone can download it.")
if st.button("Generate the PDF report", icon=":material/picture_as_pdf:", type="primary", key="grp_report"):
    with st.spinner("Writing the report..."):
        try:
            import report
            ss.report_pdf = report.build_pdf(db)
        except Exception as e:
            ss.report_pdf = None
            st.error(f"The report could not be built: {e}")
if ss.get("report_pdf"):
    st.download_button("Download group_report.pdf", ss.report_pdf, file_name="group_report.pdf", mime="application/pdf",
                       icon=":material/download:", key="grp_report_dl")
if not members:
    st.caption("The report lists the members that are in the database, so ask everyone to add their details first.")
