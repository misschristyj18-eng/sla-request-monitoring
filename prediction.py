"""Module 3 - Predictive Modeling: load the saved pipeline, predict new requests, show metrics."""
import datetime
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_processing import (CAT_FEATURES, NUM_FEATURES, FEATURES, CHOICES,
                             make_X, labeled_subset)

MODEL_PATH = "trained_model.joblib"
INFO_PATH = "model_info.json"


@st.cache_resource
def load_model():
    """Load once. The model is NOT retrained on page refresh."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_info():
    with open(INFO_PATH) as f:
        return json.load(f)


def risk_label(p: float) -> str:
    return "High" if p >= 0.80 else ("Medium" if p >= 0.50 else "Low")


def predict_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict many rows. Returns class, breach probability, risk level, missing-field count."""
    X = make_X(frame)
    proba = model.predict_proba(X)[:, list(model.classes_).index(1)]
    out = pd.DataFrame({"Predicted_Breach": (proba >= 0.5).astype(int),
                        "Delay_Risk": proba,
                        "Risk_Level": [risk_label(p) for p in proba],
                        "Missing_Fields": X.isna().sum(axis=1).values}, index=frame.index)
    return out


def top_reasons(model, row: pd.DataFrame, n=3):
    """Plain-language reasons = the split rules this request followed in the tree."""
    prep, tree = model.named_steps["prep"], model.named_steps["tree"]
    xt = prep.transform(row)
    names = prep.get_feature_names_out()
    path = tree.decision_path(xt).indices
    reasons = []
    for node in path:
        if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
            continue
        name = names[tree.tree_.feature[node]]
        thr = tree.tree_.threshold[node]
        val = xt[0, tree.tree_.feature[node]]
        val = val.item() if hasattr(val, "item") else val
        clean = name.replace("num__", "").replace("cat__", "")
        if name.startswith("num__"):
            side = "<=" if val <= thr else ">"
            reasons.append(f"{clean} {side} {thr:.1f}")
        else:
            reasons.append(f"{clean} {'= yes' if val > thr else '= no'} (one-hot rule)")
    return reasons[:n] if reasons else ["The tree made one overall decision for this case."]


def similar_rate(df: pd.DataFrame, office: str, queue_len):
    """Breach rate of similar past requests (same office and queue category)."""
    lab = labeled_subset(df)
    if queue_len is None or pd.isna(queue_len):
        sub = lab[lab["Assigned_Office"] == office]
        label = f"office = {office}"
    else:
        cat = "Low" if queue_len <= 5 else ("Medium" if queue_len <= 15 else "High")
        sub = lab[(lab["Assigned_Office"] == office) & (lab["Queue_Category"].astype(str) == cat)]
        label = f"office = {office}, queue = {cat}"
    if len(sub) == 0:
        return None, 0, label
    return float(sub["SLA_Breached"].mean()), len(sub), label


def show(df: pd.DataFrame):
    st.header("Prediction")
    model, info = load_model(), load_info()

    st.info("Advice only. The model flags requests for **human review**. It never approves, "
            "rejects or changes a request by itself.")

    tab1, tab2, tab3 = st.tabs(["Predict one request", "Predict a CSV batch", "Model metrics"])

    # ---------------- single prediction ----------------
    with tab1:
        st.caption("Only information known **at submission time** is asked for (no leakage).")
        c1, c2, c3 = st.columns(3)
        vals = {}
        vals["Service_Type"] = c1.selectbox("Service type", CHOICES["Service_Type"])
        vals["Requester_Type"] = c1.selectbox("Requester type", CHOICES["Requester_Type"])
        vals["Requested_Channel"] = c1.selectbox("Channel", CHOICES["Requested_Channel"])
        vals["Priority"] = c2.selectbox("Priority", CHOICES["Priority"], index=2)
        vals["Assigned_Office"] = c2.selectbox("Assigned office", CHOICES["Assigned_Office"])
        vals["Required_Documents_Complete"] = c2.selectbox("Required documents complete?",
                                                           CHOICES["Required_Documents_Complete"], index=1)
        vals["Submission_Weekday"] = c3.selectbox("Submission weekday", CHOICES["Submission_Weekday"])
        known_q = c3.checkbox("Queue length is known", value=True)
        q = c3.number_input("Queue length (open requests)", min_value=0, max_value=200, value=10, step=1,
                            disabled=not known_q)
        sla = c3.number_input("SLA days (target)", min_value=1, max_value=60, value=5, step=1)
        vals["Queue_Length"] = float(q) if known_q else np.nan
        vals["SLA_Days"] = float(sla)

        if st.button("Predict delay risk", type="primary"):
            row = pd.DataFrame([vals])[FEATURES]            # exact training feature names
            for c in NUM_FEATURES:
                row[c] = row[c].astype(float)
            res = predict_frame(model, row).iloc[0]
            p = float(res["Delay_Risk"])
            lab = "WILL LIKELY BREACH the SLA (1)" if res["Predicted_Breach"] == 1 else "Likely on time (0)"
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted class", lab)
            m2.metric("Breach probability", f"{p * 100:.0f}%")
            m3.metric("Risk level", res["Risk_Level"])
            st.write("**Main reasons (rules the tree followed):** " + "; ".join(top_reasons(model, row)))
            rate, n, label = similar_rate(df, vals["Assigned_Office"], None if not known_q else q)
            if rate is not None:
                st.write(f"**Similar past requests** ({label}): {rate * 100:.1f}% breached (n={n}).")
            if res["Missing_Fields"] > 0:
                st.warning("Data-completeness warning: Queue length is missing, so it was filled with the "
                           "median (a guess). This score is less reliable - check the request carefully.")
            st.caption(f"Model: {info['model_version']} | Predicted at: "
                       f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | Tree accuracy "
                       f"{info['metrics_tree']['accuracy']:.3f}, recall {info['metrics_tree']['recall']:.3f} "
                       f"(only {info['n_test']} test requests).")
            st.session_state["last_prediction"] = {"inputs": vals, "risk": p}
            st.info("Staff decision: use the **Dashboard** page to confirm or override the flag with a reason.")

    # ---------------- batch prediction ----------------
    with tab2:
        up = st.file_uploader("Upload a CSV with the 9 feature columns (or the full request file)",
                              type="csv", key="batch")
        if up is not None:
            try:
                b = pd.read_csv(up)
            except Exception as e:
                st.error(f"Could not read this file as CSV: {e}")
                b = None
            if b is not None:
                if "Submission_Weekday" not in b.columns and "Submission_Date" in b.columns:
                    b["Submission_Weekday"] = pd.to_datetime(b["Submission_Date"], errors="coerce") \
                        .dt.day_name().fillna("Unknown")
                miss = [c for c in FEATURES if c not in b.columns]
                if b.empty:
                    st.error("The file has no rows.")
                elif miss:
                    st.error("Missing column(s) needed by the model: " + ", ".join(miss))
                else:
                    for c in NUM_FEATURES:
                        b[c] = pd.to_numeric(b[c], errors="coerce")
                        b.loc[b[c] < 0, c] = np.nan
                    out = pd.concat([b, predict_frame(model, b)], axis=1)
                    st.dataframe(out.sort_values("Delay_Risk", ascending=False), use_container_width=True)
                    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                                       "predictions.csv", "text/csv")

    # ---------------- metrics ----------------
    with tab3:
        mt, mb = info["metrics_tree"], info["metrics_baseline"]
        table = pd.DataFrame([
            {"Model": "Decision Tree", "Accuracy": mt["accuracy"], "Precision (breach)": mt["precision"],
             "Recall (breach)": mt["recall"], "F1 (breach)": mt["f1"]},
            {"Model": "Majority-class baseline", "Accuracy": mb["accuracy"], "Precision (breach)": mb["precision"],
             "Recall (breach)": mb["recall"], "F1 (breach)": mb["f1"]}]).round(3)
        st.dataframe(table, use_container_width=True, hide_index=True)
        cm = info["confusion_matrix"]
        st.write("**Confusion matrix (midterm test set, 60 requests)**")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0 (on time)", "Actual 1 (breached)"],
                                  columns=["Predicted 0 (on time)", "Predicted 1 (breached)"]))
        imp = pd.Series(info["feature_importance"]).sort_values().reset_index()
        imp.columns = ["Feature", "Importance"]
        import plotly.express as px
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                               title="Feature importance (not proof of cause)"), use_container_width=True)

        if st.button("Check consistency with the midterm notebook"):
            lab = labeled_subset(df)
            if len(lab) < 50:
                st.error("Not enough labeled rows to repeat the midterm test.")
            else:
                X, y = make_X(lab), lab["SLA_Breached"]
                _, Xt, _, yt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                pr = model.predict(Xt)   # saved model, no retraining
                st.write(f"Saved model on the same 60 test rows -> accuracy {accuracy_score(yt, pr):.3f}, "
                         f"precision {precision_score(yt, pr, zero_division=0):.3f}, "
                         f"recall {recall_score(yt, pr, zero_division=0):.3f}, "
                         f"F1 {f1_score(yt, pr, zero_division=0):.3f}, "
                         f"confusion matrix {confusion_matrix(yt, pr, labels=[0, 1]).tolist()}")
                st.success("These numbers should match the midterm report (0.933 / 0.931 / 0.964 / 0.964).")

        st.subheader("Limitations")
        st.markdown(
            "- Only **297 labeled** requests and a **60-row test set** - results can change a lot.\n"
            "- **90.6%** of labeled requests breached, so the baseline already scores 0.900 accuracy.\n"
            "- The tree found only 2 of 6 on-time requests in the test set (4 false alarms).\n"
            "- Feature importance shows how the tree split data, **not** what causes delays.\n"
            "- Missing values are filled with guesses (median / most common).\n"
            "- Data is synthetic/authorized. Predictions are advice - a person decides.")
