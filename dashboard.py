"""Module 4 - Decision-Support Dashboard: KPIs, risk/review queue, interpretation, human follow-up."""
import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

from data_processing import labeled_subset
from exploratory_analysis import rate_table
from prediction import load_model, load_info, predict_frame


def show(df: pd.DataFrame, role: str):
    st.header("Decision-Support Dashboard")
    model, info = load_model(), load_info()
    lab = labeled_subset(df)

    if role == "Requester":
        st.error("Requesters can only submit and track their own requests. The risk dashboard is for "
                 "Office Staff and Administrators. (Change role in the sidebar to see it.)")
        return

    # ---- KPIs ----
    open_df = df[df["Status"].astype(str).str.lower() == "pending"].copy()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total requests", len(df))
    k2.metric("Completed requests", len(lab))
    k3.metric("SLA breach rate", f"{lab['SLA_Breached'].mean() * 100:.1f}%")
    k4.metric("Open (Pending) requests", len(open_df))

    # ---- charts ----
    c1, c2 = st.columns(2)
    t = rate_table(lab, "Assigned_Office")
    c1.plotly_chart(px.bar(t, y="Assigned_Office", x="breach_rate", orientation="h",
                           labels={"breach_rate": "Breach rate (%)"},
                           title="SLA breach rate by office (completed)"), use_container_width=True)
    q = open_df["Queue_Category"].astype(str).replace("nan", "Unknown").value_counts() \
        .reindex(["Low", "Medium", "High", "Unknown"]).dropna().reset_index()
    q.columns = ["Queue_Category", "Open requests"]
    c2.plotly_chart(px.bar(q, x="Queue_Category", y="Open requests",
                           title="Queue status: open requests by queue category at submission"),
                    use_container_width=True)

    # ---- review queue ----
    st.subheader("Flagged for human review (open requests)")
    thr = st.slider("Show requests with delay risk at or above", 0.0, 1.0, 0.50, 0.05)
    if open_df.empty:
        st.info("There are no open (Pending) requests in this dataset.")
        return
    scored = pd.concat([open_df, predict_frame(model, open_df)], axis=1)
    flagged = scored[scored["Delay_Risk"] >= thr].sort_values("Delay_Risk", ascending=False)
    flagged["Data_Warning"] = flagged["Missing_Fields"].map(lambda n: "Queue length missing" if n else "")
    st.write(f"**{len(flagged)} of {len(scored)} open requests** are at {thr * 100:.0f}% risk or higher.")
    show_cols = ["Request_ID", "Service_Type", "Assigned_Office", "Priority", "Queue_Length",
                 "Delay_Risk", "Risk_Level", "Data_Warning"]
    st.dataframe(flagged[show_cols].assign(Delay_Risk=lambda d: (d["Delay_Risk"] * 100).round(0).astype(int).astype(str) + "%"),
                 use_container_width=True, hide_index=True)

    st.info(f"Because {info['breach_share_labeled'] * 100:.1f}% of past requests breached, most open requests "
            "will be flagged. Use **Delay_Risk, queue length and priority together** to decide which to open first.")

    # ---- human decision ----
    st.subheader("Staff decision (human in the loop)")
    if role != "Office Staff":
        st.warning("Only **Office Staff** can record a final decision. Administrators can view results.")
    elif flagged.empty:
        st.info("No flagged requests to review.")
    else:
        log = st.session_state.setdefault("decision_log", [])
        rid = st.selectbox("Choose a flagged request", flagged["Request_ID"].tolist())
        choice = st.radio("Your decision", ["Confirm flag (prioritize this request)",
                                            "Override flag (not needed)"], horizontal=True)
        why = st.text_area("Reason (required)", placeholder="Example: Student has an urgent scholarship deadline.")
        if st.button("Save decision"):
            if not why.strip():
                st.error("Please write a reason before saving.")
            else:
                risk = float(flagged.loc[flagged["Request_ID"] == rid, "Delay_Risk"].iloc[0])
                log.append({"Time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", "Request_ID": rid,
                            "Model_Risk": f"{risk * 100:.0f}%", "Decision": choice.split(" (")[0],
                            "Reason": why.strip(), "Model_Version": info["model_version"], "Decided_by": role})
                st.success(f"Decision saved for {rid}. The request itself was not changed automatically.")
        if log:
            st.write("**Decision log (this session)**")
            st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
            st.download_button("Download decision log", pd.DataFrame(log).to_csv(index=False).encode(),
                               "decision_log.csv", "text/csv")

    # ---- interpretation ----
    st.subheader("How to read this dashboard")
    st.markdown(
        "- **High queue (>15) -> 100% breached** in past data; consider adding staff or sending an alert.\n"
        "- **Highest-breach office:** review workload and steps there first, then compare with the best office.\n"
        "- **Recommended human follow-up:** open the top flagged requests, check documents, talk to the requester, "
        "then confirm or override with a reason.\n"
        "- Risk scores are **advice only** (model accuracy "
        f"{info['metrics_tree']['accuracy']:.3f}, recall {info['metrics_tree']['recall']:.3f}, tested on {info['n_test']} rows).")
