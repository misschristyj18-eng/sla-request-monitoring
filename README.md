# Student Service Request Monitoring System (Streamlit)
**Baclay, Christ John S. | BSIS 3-B | Systems Analysis and Design | Section B**

A Streamlit app built from my midterm exam (Version 2). A saved Decision Tree predicts whether a service request will breach its SLA, so Office Staff can review risky requests first. Advice only - a person decides.

## Live app
- GitHub repository: https://github.com/misschristyj18-eng/sla-request-monitoring
- Streamlit app: https://sla-request-monitoring-5h66pzruadbw5j9wa5bf2k.streamlit.app/#analytical-process-decision-workflow

## Project pages
- Home
- Dataset
- Data Analysis
- Prediction
- Dashboard
- System Design

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository layout
```text
sla-request-monitoring/
├── app.py
├── requirements.txt
├── README.md
├── DEPLOYMENT_REPORT.md
├── midterm_notebook.ipynb
├── train_model.py
├── sample_service_requests.csv
├── trained_model.joblib
├── model_info.json
├── use_case_diagram.png
├── modules/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── exploratory_analysis.py
│   ├── prediction.py
│   ├── dashboard.py
│   └── system_design.py
├── models/
│   └── .gitkeep
├── assets/
│   └── .gitkeep
└── .gitignore
```

Note: the active app code is in `modules/`, while the model artifact and supporting files are stored at the repo root because the app loads them with root-relative paths such as `trained_model.joblib` and `model_info.json`.

## Model
The model is the same as the midterm:
- Decision Tree (max_depth=4, min_samples_leaf=5, random_state=42)
- 80/20 split: 237 train / 60 test
- Accuracy: 0.933
- Precision: 0.931
- Recall: 1.000
- F1: 0.964
- Leakage columns removed: Processing_Days, Status, Reopened, Satisfaction_Score

## Important notes
- The app does not retrain on refresh. The model is loaded once with `@st.cache_resource`.
- Predictions are advisory only and are not automatic approval or denial decisions.
- Synthetic data is used. No real student records are stored.
- The live app URL above is the final deployed version for verification and review.

## Final submission checklist
- [x] GitHub repository uploaded
- [x] Streamlit app deployed
- [x] App URL verified
- [x] Screenshots of the main modules captured from the live app
- [x] Five sample predictions recorded from the Prediction page
- [x] Report completed with the live app URL and summary of testing
