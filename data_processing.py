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


def validate_schema(df: pd.DataFrame):
    """Return (ok, messages). Gives a clear message for every problem."""
    msgs = []
    if df is None or df.empty:
        return False, ["The file is empty (no rows). Please upload a CSV with data."]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msgs.append("Missing required column(s): " + ", ".join(missing))
    extra = [c for c in df.columns
             if c not in REQUIRED_COLUMNS + ["Submission_Weekday", "Queue_Category", "SLA_Breached"]]
    if extra:
        msgs.append("Extra column(s) found (ignored): " + ", ".join(extra))
    return (len(missing) == 0), msgs


def quality_summary(df: pd.DataFrame):
    """Null and duplicate summary for the Dataset page."""
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].rename("Missing values").to_frame()
    nulls["% of rows"] = (nulls["Missing values"] / len(df) * 100).round(1)
    dup_id = int(df["Request_ID"].duplicated().sum()) if "Request_ID" in df.columns else 0
    dup_rows = int(df.duplicated().sum())
    return nulls, dup_id, dup_rows


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Light version of the midterm cleaning + feature engineering (Q4-Q5)."""
    df = raw.copy()
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
    done = df["Status"].astype(str).str.lower() == "completed"
    ok = done & df["Processing_Days"].notna() & df["SLA_Days"].notna()
    df["SLA_Breached"] = np.where(ok, (df["Processing_Days"] > df["SLA_Days"]).astype(int), np.nan)
    return df


def labeled_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["SLA_Breached"].notna()].copy()
    out["SLA_Breached"] = out["SLA_Breached"].astype(int)
    return out


def make_X(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact same preparation for training data and new requests (midterm make_X)."""
    X = frame[FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(object)
    return X
