"""Module 5 - SAD Integration: dynamic requirements, actors, use case diagram, decision workflow."""
import pandas as pd
import streamlit as st

from modules.data_processing import labeled_subset


def _safe_pct(value):
    return f"{value * 100:.1f}%" if pd.notna(value) else "N/A"


def _queue_bucket(value):
    if pd.isna(value):
        return "Unknown"
    if value <= 5:
        return "Low"
    if value <= 15:
        return "Medium"
    return "High"


def show(df: pd.DataFrame | None = None):
    st.header("System Design (SAD Integration)")

    if df is None or df.empty:
        st.info("No dataset is currently loaded. Upload a CSV to generate dataset-specific system design insights.")
        return

    frame = df.copy()
    if "Queue_Category" not in frame.columns and "Queue_Length" in frame.columns:
        frame["Queue_Category"] = pd.cut(
            frame["Queue_Length"],
            bins=[-float("inf"), 5, 15, float("inf")],
            labels=["Low", "Medium", "High"],
            right=False,
        )

    labeled = labeled_subset(frame)
    open_df = frame[frame["Status"].astype(str).str.lower() == "pending"].copy() if "Status" in frame.columns else pd.DataFrame()

    total_requests = len(frame)
    completed_requests = len(labeled)
    pending_requests = len(open_df)
    breach_rate = float(labeled["SLA_Breached"].mean()) if not labeled.empty else 0.0

    office_breach = pd.DataFrame()
    if not labeled.empty and "Assigned_Office" in labeled.columns and "SLA_Breached" in labeled.columns:
        office_breach = (
            labeled.groupby("Assigned_Office", observed=False)["SLA_Breached"]
            .agg(requests="count", breaches="sum", breach_rate="mean")
            .reset_index()
        )
        office_breach["breach_rate"] = office_breach["breach_rate"] * 100
        office_breach = office_breach.sort_values("breach_rate", ascending=False)

    queue_summary = pd.DataFrame()
    if not open_df.empty and "Queue_Category" in open_df.columns:
        queue_summary = (
            open_df["Queue_Category"].astype(str).value_counts().reindex(["Low", "Medium", "High"], fill_value=0).reset_index()
        )
        queue_summary.columns = ["Queue_Category", "Open requests"]

    top_office = office_breach.iloc[0] if not office_breach.empty else None
    low_office = office_breach.iloc[-1] if not office_breach.empty else None
    queue_high = 0
    if not open_df.empty and "Queue_Category" in open_df.columns:
        queue_high = int((open_df["Queue_Category"].astype(str) == "High").sum())

    st.subheader("Functional requirements (generated from the current dataset)")
    st.markdown(
        "| ID | Requirement | Current dataset evidence | Where in this app |\n"
        "|---|---|---|---|\n"
        f"| FR1 | Show SLA breach rate per assigned office | Highest office: {top_office['Assigned_Office']} at {_safe_pct(top_office['breach_rate']/100) if top_office is not None else 'N/A'} ({int(top_office['requests']) if top_office is not None else 0} requests); lowest office: {low_office['Assigned_Office']} at {_safe_pct(low_office['breach_rate']/100) if low_office is not None else 'N/A'} | Data Analysis, Dashboard |\n"
        f"| FR2 | Show open requests and queue category; alert when queue > 15 | {pending_requests} pending requests; {queue_high} are in the High queue bucket | Dashboard (queue chart) |\n"
        f"| FR3 | Calculate a delay-risk flag from submission-time data, show reasons, and send to staff for review | Current model result: {len(labeled)} labeled requests with a breach rate of {_safe_pct(breach_rate)} | Prediction, Dashboard |\n"
        "| FR4 | Let Office Staff confirm or override a flag with a written reason | Human decision step remains required before final action | Dashboard (Staff decision) |\n"
        f"| FR5 | Let users upload a CSV, validate its columns and summarize nulls/duplicates | Dataset currently in use contains {total_requests} rows, {completed_requests} completed records, and {pending_requests} pending records | Dataset |"
    )

    st.subheader("Non-functional requirements")
    st.markdown(
        "- **NFR1 Explainability:** every prediction shows risk %, main reasons, similar past requests and a limits note.\n"
        "- **NFR2 Privacy and safety:** only synthetic or authorized data; no personal IDs or passwords; no leakage columns used.\n"
        "- **NFR3 Performance and reliability:** the saved model loads once (cached) and is never retrained on refresh; invalid files show clear errors.\n"
        "- **NFR4 Adaptability:** requirements and dashboard summaries update when a new CSV is uploaded and validated."
    )

    st.subheader("Actors and permissions")
    st.markdown(
        "| Actor | View risk results | Record final decision |\n|---|---|---|\n"
        "| Requester | No (submit and track only) | No |\n"
        "| Office Staff | Yes | **Yes** (confirm or override with reason) |\n"
        "| Administrator | Yes (SLA performance, queue status) | No |"
    )

    st.subheader("Updated use case diagram")
    try:
        st.image("use_case_diagram.png", use_container_width=True)
    except Exception:
        st.error("Use case diagram not found: use_case_diagram.png")
    st.caption("Blue = analytics use cases. Grey = existing service operations. The current dataset changes the evidence shown in the design summary.")

    st.subheader("Analytical process / decision workflow")
    st.graphviz_chart(
        f"""
        digraph G {{ rankdir=LR; node [shape=box, style=rounded];
          A [label=\"1. Upload / load\\n{total_requests} request rows\"] -> B [label=\"2. Validate and clean\"]; 
          B -> C [label=\"3. Build submission-time\\nfeatures\"]; 
          C -> D [label=\"4. Saved model predicts\\nrisk for this CSV\"]; 
          D -> E [label=\"5. Dashboard shows\\n{pending_requests} pending\\n{_safe_pct(breach_rate)} breach rate\"]; 
          E -> F [label=\"6. Office Staff reviews\\ncurrent queue and exceptions\"]; 
          F -> G [label=\"7. Confirm / override\\nwith a reason\"]; 
        }}
        """
    )

    st.info(
        "This system design is dataset-aware: when a new CSV is uploaded, the evidence section, queue alerting summary, and workflow interpretation update to reflect the current file. "
        "Predictions are decision-support outputs, not automatic approval or rejection."
    )
