import pandas as pd
import streamlit as st

from modules import data_processing as dp
import modules.exploratory_analysis as exploratory_analysis
import modules.prediction as prediction
import modules.dashboard as dashboard
import modules.system_design as system_design

st.set_page_config(page_title="Student Service Request Monitoring", layout="wide")

SAMPLE = "sample_service_requests.csv"


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE)


@st.cache_data
def prep(raw):
    return dp.prepare_data(raw)


def get_raw():
    """Uploaded file if valid, otherwise the sample (midterm) dataset."""
    return st.session_state.get("raw_df", load_sample()), st.session_state.get("source", "Sample (midterm) dataset")


st.sidebar.title("SLA Monitoring")
menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
role = st.sidebar.selectbox("Your role", ["Office Staff", "Administrator", "Requester"])
st.sidebar.caption("Synthetic data only. Predictions are advice - a person decides.")

# Make role selection visible and persistent across pages.
st.session_state["role"] = role

raw, source = get_raw()

if menu == "Home":
    st.title("Student Service Request Monitoring System")
    st.caption("Baclay, Christ John S. | BSIS 3-B | Systems Analysis and Design | Section B")
    st.info(f"Current user role: {role}")

    if role == "Requester":
        st.warning("Requester mode: you can submit and track requests only. Dashboard access is restricted.")
    elif role == "Office Staff":
        st.success("Office Staff mode: you can review risk alerts and save final decisions.")
    else:
        st.info("Administrator mode: you can review SLA performance and queue status.")

    st.markdown(
        "This app turns my **midterm data-mining work** into a working system. A saved Decision Tree predicts "
        "which requests are likely to **breach their SLA** (take longer than the target days), so Office Staff "
        "can review the risky ones first.")
    a, b, c = st.columns(3)
    a.metric("Requests in data", len(raw))
    b.metric("Model", "Decision Tree")
    c.metric("Midterm F1 (breach)", f"{prediction.load_info()['metrics_tree']['f1']:.3f}")
    st.markdown(
        "**How to use:** 1) *Dataset* - upload and check a CSV. 2) *Data Analysis* - explore charts. "
        "3) *Prediction* - test one request or a batch. 4) *Dashboard* - review flagged requests and record a "
        "decision. 5) *System Design* - requirements, use case diagram, workflow.")
    st.warning("Important: the prediction is a warning only. It is never used to reject a request or deny a service.")
    st.caption(f"Data source in use: {source} | Model version: {prediction.load_info()['model_version']}")

elif menu == "Dataset":
    st.header("Dataset Upload and Validation")
    st.caption(f"Current role: {role}")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            new = pd.read_csv(up)
        except Exception as e:
            st.error(f"This file could not be read as a CSV: {e}")
            new = None
        if new is not None:
            ok, msgs = dp.validate_schema(new)
            for m in msgs:
                (st.warning if ok else st.error)(m)
            if ok:
                st.session_state["raw_df"], st.session_state["source"] = new, f"Uploaded: {up.name}"
                raw, source = new, st.session_state["source"]
                st.success("Schema check passed. This file is now used in every module.")
    else:
        st.info("No file uploaded - showing the sample (midterm) dataset.")
        if "raw_df" in st.session_state and st.button("Go back to the sample dataset"):
            st.session_state.pop("raw_df"); st.session_state.pop("source"); st.rerun()
    st.write(f"**Source:** {source}")
    st.write("Rows:", len(raw), "| Columns:", raw.shape[1])
    st.dataframe(raw.head(20), use_container_width=True)
    if not raw.empty and "Request_ID" in raw.columns:
        nulls, dup_id, dup_rows = dp.quality_summary(raw)
        c1, c2 = st.columns(2)
        c1.metric("Duplicate Request_ID rows", dup_id)
        c2.metric("Fully duplicated rows", dup_rows)
        st.write("**Missing values per column**")
        st.dataframe(nulls if len(nulls) else pd.DataFrame({"Result": ["No missing values"]}), use_container_width=True)
        st.caption("Missing Processing_Days / Satisfaction_Score / Reopened are normal for open or cancelled "
                   "requests. Missing Queue_Length is filled inside the model pipeline.")

else:
    ok, msgs = dp.validate_schema(raw)
    if not ok:
        st.error("The current data is not valid. " + " ".join(msgs) + " Go to the Dataset page.")
    elif menu == "Data Analysis":
        st.caption(f"Current role: {role}")
        exploratory_analysis.show(prep(raw))
    elif menu == "Prediction":
        st.caption(f"Current role: {role}")
        prediction.show(prep(raw))
    elif menu == "Dashboard":
        st.caption(f"Current role: {role}")
        dashboard.show(prep(raw), role)
    elif menu == "System Design":
        st.caption(f"Current role: {role}")
        system_design.show()
