# Student Service Request Monitoring System (Streamlit)
**Baclay, Christ John S. | BSIS 3-B | Systems Analysis and Design | Section B**

A Streamlit app built from my midterm exam (Version 2). A saved Decision Tree predicts whether a service
request will **breach its SLA**, so Office Staff can review risky requests first. **Advice only - a person decides.**

## Pages
Home | Dataset (upload + validation) | Data Analysis (EDA) | Prediction | Dashboard | System Design

## Run on your computer
```
pip install -r requirements.txt
streamlit run app.py
```
(`python train_model.py` re-creates `models/trained_model.joblib` from the midterm code. The app itself never retrains.)

## Files
- `app.py` - navigation | `modules/` - data_processing, exploratory_analysis, prediction, dashboard, system_design
- `models/trained_model.joblib` + `model_info.json` - saved pipeline and midterm metrics
- `assets/use_case_diagram.png` - updated UML use case diagram
- `data/sample_service_requests.csv` - synthetic midterm data (no real student records)
- `midterm_notebook.ipynb` - original midterm notebook | `train_model.py` - midterm model code, saved once

## Model (same as midterm)
Decision Tree (max_depth=4, min_samples_leaf=5, random_state=42), 80/20 split (237 train / 60 test).
Accuracy 0.933, Precision 0.931, Recall 1.000, F1 0.964 (baseline F1 0.947).
Leakage columns removed: Processing_Days, Status, Reopened, Satisfaction_Score.
