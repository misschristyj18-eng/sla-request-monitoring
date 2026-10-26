# Short Deployment Report
**Name:** Baclay, Christ John S. | **Section:** BSIS 3-B | **Course:** Systems Analysis and Design
**Project:** Student Service Request Monitoring (Section B) | **GitHub URL:** _paste here_ | **Streamlit URL:** _paste here_

## 1. Reused code (from the midterm notebook)
- Same cleaning and feature steps (duplicate IDs removed, negative numbers made missing, Submission_Weekday, Queue_Category, SLA_Breached).
- Same 9 features, same target, same 80/20 stratified split (random_state=42), same pipeline (median / most-frequent fill, one-hot, Decision Tree depth 4, leaf 5).
- Same leakage rule: an `assert` stops the app if Processing_Days, Status, Reopened or Satisfaction_Score is used as a feature.
- Same charts idea: breach rate by queue category and by office; same wireframe KPIs.

## 2. Changes made
- Model saved once with joblib (`models/trained_model.joblib`) and loaded with `@st.cache_resource`; it is not retrained on refresh.
- Notebook code split into modules (`data_processing`, `exploratory_analysis`, `prediction`, `dashboard`, `system_design`).
- Matplotlib charts became interactive Plotly charts.
- Added: CSV upload with schema check, single and batch prediction, probability + risk level, reasons, missing-data warning, staff Confirm/Override with a required reason, and a decision log.
- Added role selector (Requester / Office Staff / Administrator): only Office Staff record decisions.
- Requirements FR1-FR3 kept; FR4 (staff confirm/override) and FR5 (CSV upload and validation) were added to match features that now exist.

## 3. Tests
| Test | Result |
|---|---|
| Saved model on midterm test rows | Accuracy 0.933, precision 0.931, recall 1.000, F1 0.964, confusion matrix [[2,4],[0,54]] - **matches midterm** |
| Feature importance | Queue_Length 0.63, Required_Documents_Complete 0.30, Service_Type 0.04 - **matches midterm** |
| Sample CSV validation | Passes (400 rows, 14 required columns) |
| Empty CSV | Clear error: "The file is empty" |
| CSV with a missing column | Clear error naming the missing column |
| Non-CSV / unreadable file | Error message, app does not crash |
| Open requests flagged at 50% risk | 79 of 81 (same as the midterm wireframe) |
| Unknown queue length | Prediction still works; warning shown |

**Five sample predictions** (Records, Student, Portal, Normal, Registrar, Monday, SLA 5 days; only the listed items change):

| # | Case | Predicted | Breach probability | Risk level | Missing-data warning |
|---|---|---|---|---|---|
| 1 | Queue 0, docs complete | On time (0) | 0% | Low | No |
| 2 | Queue 1, docs complete | On time (0) | 43% | Low | No |
| 3 | Queue 3, docs complete | Breach (1) | 80% | High | No |
| 4 | Queue 20, docs NOT complete | Breach (1) | 100% | High | No |
| 5 | Queue unknown, docs complete | Breach (1) | 98% | High | Yes - queue filled with median |

> Screenshots of the five modules and these five predictions go in Section 6 (take them from your deployed app).

## 4. Limitations
- Only 297 labeled requests and a 60-row test set - results can change a lot.
- 90.6% of requests breached, so the baseline already scores 0.900 accuracy; the tree found only 2 of 6 on-time requests.
- Because most cases are "breach", most open requests get flagged; staff should use risk %, queue length and priority together.
- Feature importance is not proof of cause. Missing values are guesses (median / most common).
- Data is synthetic. The decision log lives only in the session (it resets on refresh) unless downloaded.
- Role selection is a demo, not real login security.

## 5. Reflection
- **Why human review, not automatic denial:** the model learns only from past requests and can be wrong; it does not know urgent cases. The flag only says "look at this first"; staff decide.
- **Missing data:** missing values are filled with guesses, so the score is less reliable. The app shows a warning and staff should check the request more carefully.
- **What explains a prediction:** risk % and level, main reasons, breach rate of similar requests, missing-data warning, a note with accuracy/recall, and buttons to confirm or override with a reason.
- **Lesson learned:** turning a notebook into an app means saving the model once, matching feature names exactly, and showing limits - not only the score.

## 6. Screenshots (paste here)
Home | Dataset | Data Analysis | Prediction (5 samples) | Dashboard | System Design
