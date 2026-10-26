"""Module 4 - Decision-Support Dashboard: KPIs, risk/review queue, interpretation, human follow-up."""
import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import labeled_subset
from modules.exploratory_analysis import rate_table
from modules.prediction import load_info, load_model, predict_frame


st.markdown(
    """
    <style>
    .kpi-box {
        background-color: #f0f0f0;
        border: 1px solid #3b3b3b;
        padding: 18px 16px 12px 16px;
        border-radius: 0px;
        text-align: center;
        min-height: 120px;
    }
    .kpi-box .value {
        font-size: 2rem;
        font-weight: 700;
        color: #111111;
        line-height: 1.2;
    }
    .kpi-box .label {
        font-size: 1rem;
        color: #111111;
        margin-top: 0.25rem;
    }
    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SERVICE_COLORS = ["#1f78b4", "#2f7fbf", "#4d9bd3"]
OFFICE_COLORS = ["#f5a25d", "#eb6a3d", "#d9532f"]
QUEUE_COLORS = ["#b7d9e8", "#5fa6d9", "#0d4e87"]


def risk_bg(value: int) -> str:
    if value >= 80:
        return "background-color: #d32f2f; color: white; font-weight: 700;"
    if value >= 60:
        return "background-color: #f57c00; color: white; font-weight: 700;"
    if value >= 40:
        return "background-color: #fdd835; color: #111111; font-weight: 700;"
    return "background-color: #2e7d32; color: white; font-weight: 700;"


def show(df: pd.DataFrame, role: str):
    st.markdown('<div class="dashboard-title">Administrator Dashboard - WIREFRAME (sample made from the dataset)</div>', unsafe_allow_html=True)

    model, info = load_model(), load_info()
    lab = labeled_subset(df)

    if role == "Requester":
        st.error("Requesters can only submit and track their own requests. The risk dashboard is for Office Staff and Administrators. (Change role in the sidebar to see it.)")
        return

    open_df = df[df["Status"].astype(str).str.lower() == "pending"].copy()

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="value">{len(df):,}</div>
                <div class="label">Total requests</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="value">{len(lab):,}</div>
                <div class="label">Completed requests</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="value">{lab['SLA_Breached'].mean() * 100:.1f}%</div>
                <div class="label">SLA breach rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="value">{len(open_df):,}</div>
                <div class="label">Open (Pending) requests</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Left / right chart layout
    c1, c2 = st.columns(2)

    # Office breach rate (orange/red palette)
    office_df = rate_table(lab, "Assigned_Office")
    office_fig = px.bar(
        office_df,
        y="Assigned_Office",
        x="breach_rate",
        orientation="h",
        labels={"breach_rate": "Breach rate (%)", "Assigned_Office": "Office"},
        color_discrete_sequence=OFFICE_COLORS,
        template="simple_white",
    )
    office_fig.update_layout(
        title="SLA breach rate by office (completed)",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Breach rate (%)"),
        yaxis=dict(title="Office"),
        showlegend=False,
    )
    c1.plotly_chart(office_fig, use_container_width=True)

    # Queue status (blue palette)
    q = (
        open_df["Queue_Category"]
        .astype(str)
        .replace("nan", "Unknown")
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .rename_axis("Queue_Category")
        .reset_index(name="Open requests")
    )
    qplot = px.bar(
        q,
        x="Queue_Category",
        y="Open requests",
        color_discrete_sequence=QUEUE_COLORS,
        template="simple_white",
    )
    qplot.update_layout(
        title="Queue status: open requests by queue category at submission",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Queue category"),
        yaxis=dict(title="Number of requests"),
        showlegend=False,
    )
    c2.plotly_chart(qplot, use_container_width=True)

    # Service volume chart (blue palette)
    s1, s2 = st.columns(2)
    service_counts = df["Service_Type"].value_counts().sort_values()
    service_counts_df = service_counts.rename_axis("Service type").reset_index(name="Number of requests")
    svc_fig = px.bar(
        service_counts_df,
        x="Number of requests",
        y="Service type",
        orientation="h",
        color_discrete_sequence=SERVICE_COLORS,
        template="simple_white",
    )
    svc_fig.update_layout(
        title="Service volume (all requests)",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Number of requests"),
        yaxis=dict(title="Service type"),
        showlegend=False,
    )
    s1.plotly_chart(svc_fig, use_container_width=True)

    # Human review table with threshold slider and color-coded risk
    s2.subheader("Flagged for human review")
    thr = st.slider("Show requests with delay risk at or above", 0.0, 1.0, 0.50, 0.05)
    if open_df.empty:
        s2.info("There are no open (Pending) requests in this dataset.")
        return

    scored = pd.concat([open_df, predict_frame(model, open_df)], axis=1)
    flagged = scored[scored["Delay_Risk"] >= thr].sort_values("Delay_Risk", ascending=False).copy()
    flagged["Data_Warning"] = flagged["Missing_Fields"].map(lambda n: "Queue length missing" if n else "")

    s2.write(f"**{len(flagged)} of {len(scored)} open requests** are at {thr * 100:.0f}% risk or higher.")

    display_df = flagged[["Request_ID", "Service_Type", "Assigned_Office", "Delay_Risk", "Risk_Level"]].copy()
    display_df["Delay_Risk"] = (display_df["Delay_Risk"] * 100).round(0).astype(int)
    display_df.columns = ["Request_ID", "Service", "Office", "Delay risk (%)", "Risk level"]

    styled = display_df.style.apply(
        lambda row: [
            risk_bg(row["Delay risk (%)"]) if col == "Delay risk (%)" else "" for col in row.index
        ],
        axis=1,
    )
    s2.dataframe(styled, use_container_width=True, hide_index=True)

    st.write("")
    st.info("Advice only - staff decide. [Review] [Override with reason]")

    # Staff decision section
    st.subheader("Staff decision (human in the loop)")
    if role != "Office Staff":
        st.warning("Only Office Staff can record a final decision. Administrators can view results.")
    elif flagged.empty:
        st.info("No flagged requests to review.")
    else:
        log = st.session_state.setdefault("decision_log", [])
        rid = st.selectbox("Choose a flagged request", flagged["Request_ID"].tolist())
        choice = st.radio("Your decision", ["Confirm flag (prioritize this request)", "Override flag (not needed)"], horizontal=True)
        why = st.text_area("Reason (required)", placeholder="Example: Student has an urgent scholarship deadline.")
        if st.button("Save decision"):
            if not why.strip():
                st.error("Please write a reason before saving.")
            else:
                risk = float(flagged.loc[flagged["Request_ID"] == rid, "Delay_Risk"].iloc[0])
                log.append({
                    "Time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                    "Request_ID": rid,
                    "Model_Risk": f"{risk * 100:.0f}%",
                    "Decision": choice.split(" (")[0],
                    "Reason": why.strip(),
                    "Model_Version": info["model_version"],
                    "Decided_by": role,
                })
                st.success(f"Decision saved for {rid}. The request itself was not changed automatically.")
        if log:
            st.write("**Decision log (this session)")
            st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
            st.download_button("Download decision log", pd.DataFrame(log).to_csv(index=False).encode(), "decision_log.csv", "text/csv")

    st.subheader("How to read this dashboard")
    st.markdown(
        "- High queue (>15) -> 100% breached in past data; consider adding staff or sending an alert.\n"
        "- Highest-breach office: review workload and steps there first, then compare with the best office.\n"
        "- Recommended human follow-up: open the top flagged requests, check documents, talk to the requester, then confirm or override with a reason.\n"
        f"- Risk scores are advice only (model accuracy {info['metrics_tree']['accuracy']:.3f}, recall {info['metrics_tree']['recall']:.3f}, tested on {info['n_test']} rows)."
    )

    st.caption("Synthetic data only. Predictions are advice - a person decides.")
