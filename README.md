# Student Service Request Monitoring System (Streamlit)
**Baclay, Christ John S. | BSIS 3-B | Systems Analysis and Design | Section B**

A Streamlit app built from my midterm exam (Version 2). A saved Decision Tree predicts whether a service request will breach its SLA, so Office Staff can review risky requests first. Advice only - a person decides.

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
This repo is organized for the final exam deliverable, with app logic in the `modules` package and a saved model in `models`.

```text
sla-request-monitoring/
├── app.py
├── requirements.txt
├── README.md
├── DEPLOYMENT_REPORT.md
├── midterm_notebook.ipynb
├── train_model.py
├── sample_service_requests.csv
├── models/
│   ├── trained_model.joblib
│   └── model_info.json
├── modules/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── exploratory_analysis.py
│   ├── prediction.py
│   ├── dashboard.py
│   └── system_design.py
├── assets/
│   └── use_case_diagram.png
└── data/
    └── sample_service_requests.csv
```

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

## Deployment details
- GitHub repository: https://github.com/misschristyj18-eng/sla-request-monitoring
- Streamlit deployment URL: to be filled in after final public deployment
