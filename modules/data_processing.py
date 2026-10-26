"""Data loading, validation and the SAME feature preparation used in the midterm notebook."""
import numpy as np
import pandas as pd

# ---- Same lists as the midterm notebook (Question 8) ----
CAT_FEATURES = ["Service_Type", "Requester_Type", "Requested_Channel", "Priority",
                "Assigned_Office", "Required_Documents_Complete", "Submission_Weekday"]
NUM_FEATURES = ["Queue_Length", "SLA_Days"]
FEATURES = CAT_FEATURES + NUM_FEATURES
FORBIDDEN = {"Processing_Days", "Status", "Reopened", "Satisfaction_Score", "SLA_Breached"}

# Columns a user CSV must have (the exam-paper schema)
REQUIRED_COLUMNS = ["Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
                   "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
                   "Required_Documents_Complete", "Processing_Days", "SLA_Days",
                   "Status", "Reopened", "Satisfaction_Score"]
OPTIONAL_COLUMNS = ["Submission_Weekday", "Queue_Category", "SLA_Breached"]

COLUMN_ALIASES = {
    "request_id": "Request_ID",
    "requestid": "Request_ID",
    "service_type": "Service_Type",
    "service": "Service_Type",
    "requester_type": "Requester_Type",
    "requester": "Requester_Type",
    "submission_date": "Submission_Date",
    "date_submitted": "Submission_Date",
    "submitted_date": "Submission_Date",
    "requested_channel": "Requested_Channel",
    "channel": "Requested_Channel",
    "priority": "Priority",
    "assigned_office": "Assigned_Office",
    "office": "Assigned_Office",
    "queue_length": "Queue_Length",
    "queue": "Queue_Length",
    "required_documents_complete": "Required_Documents_Complete",
    "required_documents": "Required_Documents_Complete",
    "processing_days": "Processing_Days",
    "sla_days": "SLA_Days",
    "status": "Status",
    "reopened": "Reopened",
    "satisfaction_score": "Satisfaction_Score",
    "submission_weekday": "Submission_Weekday",
    "weekday": "Submission_Weekday",
    "queue_category": "Queue_Category",
    "sla_breached": "SLA_Breached",
}

assert FORBIDDEN.isdisjoint(FEATURES), "A forbidden (after-submission) column is in the features!"

CHOICES = {
    "Service_Type": ["Certification", "Enrollment", "IT Support", "Records", "Scholarship Inquiry"],
    "Requester_Type": ["Faculty", "Staff", "Student"],
    "Requested_Channel": ["Email", "Portal", "Walk-in"],
    "Priority": ["High", "Low", "Normal"],
    "Assigned_Office": ["Dean's Office", "IT Office", "Registrar", "Scholarship Office"],
    "Required_Documents_Complete": ["No", "Yes"],
    "Submission_Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names from common CSV variations used by students."""
    norm = df.copy()
    rename_map = {}
    for col in list(norm.columns):
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        key = key.replace("'", "").replace("(", "").replace(")", "")
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    if rename_map:
        norm = norm.rename(columns=rename_map)
    return norm


def validate_schema(df: pd.DataFrame):
    """Return (ok, messages). Accepts common naming variations and auto-fills missing support columns."""
    msgs = []
    if df is None or df.empty:
        return False, ["The file is empty (no rows). Please upload a CSV with data."]

    norm = normalize_columns(df)

    critical = [
        "Request_ID", "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
        "Assigned_Office", "Queue_Length", "Required_Documents_Complete", "SLA_Days", "Status"
    ]
    missing = [c for c in critical if c not in norm.columns]
    if missing:
        msgs.append("Missing required column(s): " + ", ".join(missing))

    optional_missing = [c for c in ["Submission_Date", "Processing_Days", "Reopened", "Satisfaction_Score"] if c not in norm.columns]
    if optional_missing:
        msgs.append("The file is missing optional columns that will be filled automatically: " + ", ".join(optional_missing))

    extra = [c for c in norm.columns if c not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    if extra:
        msgs.append("Extra column(s) found (ignored): " + ", ".join(extra))

    return (len(missing) == 0), msgs


def quality_summary(df: pd.DataFrame):
    """Null and duplicate summary for the Dataset page."""
    df = normalize_columns(df)
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].rename("Missing values").to_frame()
    nulls["% of rows"] = (nulls["Missing values"] / len(df) * 100).round(1)
    dup_id = int(df["Request_ID"].duplicated().sum()) if "Request_ID" in df.columns else 0
    dup_rows = int(df.duplicated().sum())
    return nulls, dup_id, dup_rows


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Light version of the midterm cleaning + feature engineering (Q4-Q5)."""
    df = normalize_columns(raw.copy())

    if "Request_ID" not in df.columns:
        df["Request_ID"] = [f"REQ-{i + 1}" for i in range(len(df))]
    if "Submission_Date" not in df.columns:
        df["Submission_Date"] = pd.to_datetime("today").normalize() - pd.to_timedelta(range(len(df)), unit="D")
    if "Status" not in df.columns:
        df["Status"] = "Pending"
    if "Reopened" not in df.columns:
        df["Reopened"] = 0
    if "Processing_Days" not in df.columns:
        df["Processing_Days"] = np.nan
    if "Satisfaction_Score" not in df.columns:
        df["Satisfaction_Score"] = np.nan
    if "SLA_Days" not in df.columns:
        df["SLA_Days"] = 5
    if "Required_Documents_Complete" not in df.columns:
        df["Required_Documents_Complete"] = "Yes"
    if "Queue_Length" not in df.columns:
        df["Queue_Length"] = 0

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)

    df = df[~(df["Request_ID"].notna() & df["Request_ID"].duplicated(keep="first"))].copy()

    for col in ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    df.loc[df["SLA_Days"] == 0, "SLA_Days"] = np.nan
    df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
    df["Submission_Weekday"] = df["Submission_Date"].dt.day_name().fillna("Unknown")

    df["Queue_Category"] = pd.cut(df["Queue_Length"], bins=[-np.inf, 5, 15, np.inf],
                                  labels=["Low", "Medium", "High"])

    df["Status"] = df["Status"].astype(str).str.lower()
    done = df["Status"] == "completed"
    ok = done & df["Processing_Days"].notna() & df["SLA_Days"].notna()
    df["SLA_Breached"] = np.where(ok, (df["Processing_Days"] > df["SLA_Days"]).astype(int), np.nan)

    return df


def labeled_subset(df: pd.DataFrame) -> pd.DataFrame:
    if "SLA_Breached" not in df.columns:
        return pd.DataFrame()
    out = df[df["SLA_Breached"].notna()].copy()
    out["SLA_Breached"] = out["SLA_Breached"].astype(int)
    return out


def make_X(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact same preparation for training data and new requests (midterm make_X)."""
    frame = normalize_columns(frame)
    X = frame.reindex(columns=FEATURES).copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(object)
    return X
