import pandas as pd
import streamlit as st
from pathlib import Path

from modules import data_processing as dp
import modules.exploratory_analysis as exploratory_analysis
import modules.prediction as prediction
import modules.dashboard as dashboard
import modules.system_design as system_design

st.set_page_config(page_title="Student Service Request Monitoring", layout="wide")

SAMPLE = "sample_service_requests.csv"
REQUESTER_LOG = "requester_requests.csv"


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE)


@st.cache_data
def prep(raw):
    return dp.prepare_data(raw)


def load_requester_requests():
    if "requester_requests" in st.session_state:
        return st.session_state["requester_requests"]

    path = Path(REQUESTER_LOG)
    if path.exists():
        try:
            df = pd.read_csv(path)
            if not df.empty:
                records = df.to_dict(orient="records")
                st.session_state["requester_requests"] = records
                return records
        except Exception:
            pass

    st.session_state["requester_requests"] = []
    return st.session_state["requester_requests"]


def save_requester_requests(records):
    st.session_state["requester_requests"] = records
    pd.DataFrame(records).to_csv(REQUESTER_LOG, index=False)


def requester_portal():
    st.header("Requester Portal")
    st.caption("Submit new service requests and track current request status.")

    requests = load_requester_requests()

    if st.session_state.get("role") != "Requester":
        st.info("Switch your role to Requester to submit or review your own service requests.")

    if st.session_state.get("role") == "Requester":
        with st.form("requester_submit_form"):
            c1, c2, c3 = st.columns(3)
            req_id = c1.text_input("Request ID", value=f"REQ-{len(requests) + 1:04d}")
            service_type = c2.selectbox("Service type", dp.CHOICES["Service_Type"])
            requester_type = c3.selectbox("Requester type", dp.CHOICES["Requester_Type"])

            c4, c5, c6 = st.columns(3)
            channel = c4.selectbox("Requested channel", dp.CHOICES["Requested_Channel"])
            priority = c5.selectbox("Priority", dp.CHOICES["Priority"], index=2)
            office = c6.selectbox("Assigned office", dp.CHOICES["Assigned_Office"])

            c7, c8 = st.columns(2)
            queue_length = c7.number_input("Queue length", min_value=0, max_value=200, value=5, step=1)
            sla_days = c8.number_input("SLA days", min_value=1, max_value=60, value=5, step=1)

            docs_complete = st.selectbox(
                "Required documents complete?",
                dp.CHOICES["Required_Documents_Complete"],
                index=1,
            )
            submission_date = st.date_input("Submission date", value=pd.Timestamp.today().normalize())

            submitted = st.form_submit_button("Submit request", type="primary")

            if submitted:
                cleaned_id = str(req_id).strip()
                if not cleaned_id:
                    st.error("Request ID is required.")
                elif any(str(r.get("Request_ID", "")).strip().lower() == cleaned_id.lower() for r in requests):
                    st.error(f"A request with ID {cleaned_id} already exists.")
                else:
                    new_request = {
                        "Request_ID": cleaned_id,
                        "Service_Type": service_type,
                        "Requester_Type": requester_type,
                        "Submission_Date": str(submission_date),
                        "Requested_Channel": channel,
                        "Priority": priority,
                        "Assigned_Office": office,
                        "Queue_Length": int(queue_length),
                        "Required_Documents_Complete": docs_complete,
                        "Processing_Days": "",
                        "SLA_Days": int(sla_days),
                        "Status": "Pending",
                        "Reopened": 0,
                        "Satisfaction_Score": "",
                        "Submission_Weekday": submission_date.strftime("%A"),
                        "Queue_Category": "Low" if queue_length <= 5 else "Medium" if queue_length <= 15 else "High",
                        "SLA_Breached": "",
                    }
                    save_requester_requests(requests + [new_request])
                    st.success(f"Request {cleaned_id} submitted successfully.")
                    st.balloons()

    if not requests:
        st.info("No requests have been submitted yet.")
        return

    st.subheader("My request status")
    req_df = pd.DataFrame(requests)
    if req_df.empty:
        st.info("No requests available to display.")
        return

    if "Submission_Date" in req_df.columns:
        req_df["Submission_Date"] = pd.to_datetime(req_df["Submission_Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "Queue_Length" in req_df.columns:
        req_df["Queue_Length"] = pd.to_numeric(req_df["Queue_Length"], errors="coerce")
        req_df["Queue_Category"] = pd.cut(
            req_df["Queue_Length"],
            bins=[-float("inf"), 5, 15, float("inf")],
            labels=["Low", "Medium", "High"],
            right=False,
        )

    display_cols = [
        "Request_ID",
        "Service_Type",
        "Requester_Type",
        "Submission_Date",
        "Assigned_Office",
        "Priority",
        "Queue_Category",
        "Status",
        "SLA_Days",
    ]
    display_cols = [c for c in display_cols if c in req_df.columns]
    st.dataframe(req_df[display_cols], use_container_width=True, hide_index=True)


def get_raw():
    """Uploaded file if valid, otherwise the sample (midterm) dataset."""
    return st.session_state.get("raw_df", load_sample()), st.session_state.get("source", "Sample (midterm) dataset")


st.sidebar.title("SLA Monitoring")
menu = st.sidebar.selectbox(
    "Module",
    ["Home", "Dataset", "Data Analysis", "Prediction", "Requester Portal", "Dashboard", "System Design"],
)
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
        "3) *Prediction* - test one request or a batch. 4) *Requester Portal* - submit and track your requests. "
        "5) *Dashboard* - review flagged requests and record a decision. 6) *System Design* - requirements, use case diagram, workflow.")
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
    elif menu == "Requester Portal":
        st.caption(f"Current role: {role}")
        requester_portal()
    elif menu == "Dashboard":
        st.caption(f"Current role: {role}")
        dashboard.show(prep(raw), role)
    elif menu == "System Design":
        st.caption(f"Current role: {role}")
        system_design.show()
