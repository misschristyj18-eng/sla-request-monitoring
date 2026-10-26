# Short Deployment Report
**Name:** Baclay, Christ John S. | **Section:** BSIS 3-B | **Course:** Systems Analysis and Design
**Project:** Student Service Request Monitoring (Section B) | **GitHub URL:** https://github.com/misschristyj18-eng/sla-request-monitoring | **Streamlit URL:** pending final public deployment

## 1. Reused code (from the midterm notebook)
- Same cleaning and feature steps: duplicate IDs removed, negative numbers made missing, Submission_Weekday, Queue_Category, and SLA_Breached created.
- Same 9 features and target as the midterm work.
- Same leakage rule: Processing_Days, Status, Reopened, and Satisfaction_Score are not used as model features.
- Same overall model design: Decision Tree with the same midterm split and parameters.

## 2. Changes made
- Model saved once with joblib and loaded with `@st.cache_resource`; it is not retrained on refresh.
- Notebook logic was reorganized into modules: `data_processing`, `exploratory_analysis`, `prediction`, `dashboard`, and `system_design`.
- Charting moved from static plots to interactive Plotly views.
- Added CSV upload with validation, batch prediction, role-aware access logic, decision log, and system design documentation.
- Added role-based behavior aligning with the UML: Requester, Office Staff, and Administrator.

## 3. Tests
| Test | Result |
|---|---|
| Saved model on midterm test rows | Accuracy 0.933, precision 0.931, recall 1.000, F1 0.964, confusion matrix [[2,4],[0,54]] |
| Feature importance | Matches the midterm logic |
| Sample CSV validation | Passes |
| Empty CSV | Clear validation message |
| Missing column CSV | Clear column-name error |
| Unexpected file type | Safe error without crashing |
| Role switching | Access behavior changes according to the UML actor mode |

## 4. Five sample predictions
These are the sample prediction scenarios for the final deployment report.

| # | Case | Predicted | Breach probability | Risk level | Missing-data warning |
|---|---|---|---|---|---|
| 1 | Queue 0, docs complete | On time (0) | 0% | Low | No |
| 2 | Queue 1, docs complete | On time (0) | 43% | Low | No |
| 3 | Queue 3, docs complete | Breach (1) | 80% | High | No |
| 4 | Queue 20, docs NOT complete | Breach (1) | 100% | High | No |
| 5 | Queue unknown, docs complete | Breach (1) | 98% | High | Yes |

## 5. Limitations
- The data is synthetic and approved for demonstration only.
- The decision log is session-based unless downloaded.
- The model is explanatory and advisory; it does not automatically reject or deny a service.
- Missing data is filled with approximation, so staff review remains necessary.

## 6. Reflection
This project shows how a midterm machine-learning notebook can be turned into a decision-support information system by preserving the saved model, validating uploaded data, and centering human review. The dashboard is designed to help staff prioritize urgent cases without automating final service decisions.

## 7. Final public deployment status
The app code is ready and the repo is prepared for deployment. The final public Streamlit URL and screenshot proof must be added after the app is published to Streamlit Cloud.
