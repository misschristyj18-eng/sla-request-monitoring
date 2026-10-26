"""Module 2 - Exploratory Data Analysis (descriptive stats, class distribution, interactive charts)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import labeled_subset


def rate_table(labeled: pd.DataFrame, by: str) -> pd.DataFrame:
    t = (labeled.groupby(by, observed=False)["SLA_Breached"]
         .agg(requests="count", breaches="sum", breach_rate="mean").reset_index())
    t["breach_rate"] = (t["breach_rate"] * 100).round(1)
    return t.dropna(subset=["breach_rate"]).sort_values("breach_rate", ascending=False)


def show(df: pd.DataFrame):
    st.header("Data Analysis")
    labeled = labeled_subset(df)

    st.subheader("1. Descriptive statistics")
    num_cols = [c for c in ["Queue_Length", "SLA_Days", "Processing_Days", "Satisfaction_Score"] if c in df.columns]
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
    st.caption("Processing_Days and Satisfaction_Score exist only for finished requests. "
               "They are for analysis only and are never used to predict.")

    st.subheader("2. Class distribution (the target)")
    c1, c2 = st.columns(2)
    dist = labeled["SLA_Breached"].map({0: "On time (0)", 1: "Breached (1)"}).value_counts().reset_index()
    dist.columns = ["Class", "Requests"]
    c1.plotly_chart(px.pie(dist, names="Class", values="Requests", title="SLA outcome (completed requests)"),
                    use_container_width=True)
    c2.metric("Labeled (completed) requests", len(labeled))
    c2.metric("Breach share", f"{labeled['SLA_Breached'].mean() * 100:.1f}%")
    st.warning("The classes are imbalanced: most requests breached the SLA, so a model can look accurate "
               "by always saying 'breach'. Always read precision, recall and F1 too.")

    st.subheader("3. Interactive charts")
    t1, t2, t3, t4 = st.tabs(["Breach rate by queue category", "Breach rate by office",
                              "Service volume", "Queue length vs SLA days"])
    with t1:
        t = rate_table(labeled, "Queue_Category")
        t["Queue_Category"] = t["Queue_Category"].astype(str)
        st.plotly_chart(px.bar(t, x="Queue_Category", y="breach_rate", text="breach_rate",
                               hover_data=["requests", "breaches"],
                               category_orders={"Queue_Category": ["Low", "Medium", "High"]},
                               labels={"breach_rate": "Breach rate (%)"},
                               title="SLA breach rate by queue category (Low 0-5, Medium 6-15, High >15)"),
                        use_container_width=True)
        st.caption("Requests sent into longer queues breach more often. This is a link, not proof of cause.")
    with t2:
        t = rate_table(labeled, "Assigned_Office")
        st.plotly_chart(px.bar(t, y="Assigned_Office", x="breach_rate", orientation="h", text="breach_rate",
                               hover_data=["requests", "breaches"],
                               labels={"breach_rate": "Breach rate (%)"},
                               title="SLA breach rate by assigned office"), use_container_width=True)
    with t3:
        office = st.selectbox("Filter by office", ["All"] + sorted(df["Assigned_Office"].dropna().unique()))
        d = df if office == "All" else df[df["Assigned_Office"] == office]
        v = d["Service_Type"].value_counts().reset_index()
        v.columns = ["Service_Type", "Requests"]
        st.plotly_chart(px.bar(v, x="Requests", y="Service_Type", orientation="h",
                               title="Number of requests per service type"), use_container_width=True)
    with t4:
        sc = labeled.copy()
        sc["Outcome"] = sc["SLA_Breached"].map({0: "On time", 1: "Breached"})
        st.plotly_chart(px.strip(sc, x="Queue_Category", y="SLA_Days", color="Outcome",
                                 category_orders={"Queue_Category": ["Low", "Medium", "High"]},
                                 title="SLA days by queue category and outcome"), use_container_width=True)
