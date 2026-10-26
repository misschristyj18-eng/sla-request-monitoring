"""Re-creates the MIDTERM model exactly (same split, same pipeline, same settings) and saves it once.
Run:  python train_model.py     (the app never retrains - it only loads the saved file)"""
import datetime
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from modules.data_processing import CAT_FEATURES, FEATURES, NUM_FEATURES, labeled_subset, make_X, prepare_data

RANDOM_STATE = 42
raw = pd.read_csv("sample_service_requests.csv")
labeled = labeled_subset(prepare_data(raw))
X, y = make_X(labeled), labeled["SLA_Breached"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

preprocess = ColumnTransformer([
    ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
    ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
])
model = Pipeline([("prep", preprocess),
                  ("tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                                  random_state=RANDOM_STATE))])
model.fit(X_train, y_train)

pred = model.predict(X_test)
base = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(X_train), 1)), y_train)
bpred = base.predict(np.zeros((len(X_test), 1)))


def ev(p):
    return {
        "accuracy": accuracy_score(y_test, p),
        "precision": precision_score(y_test, p, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, p, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, p, pos_label=1, zero_division=0),
    }

# feature importance (one-hot columns added back to the original feature)
names = model.named_steps["prep"].get_feature_names_out()
agg = {}
for n, v in zip(names, model.named_steps["tree"].feature_importances_):
    key = n[5:] if n.startswith("num__") else next(
        (c for c in CAT_FEATURES if n[5:].startswith(c + "_")), n[5:]
    )
    agg[key] = agg.get(key, 0) + float(v)

info = {
    "model_name": "Decision Tree (max_depth=4, min_samples_leaf=5, random_state=42)",
    "model_version": "v1.0-midterm-examV2",
    "trained_on": datetime.date.today().isoformat(),
    "sklearn_version": sklearn.__version__,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "features": FEATURES,
    "target": "SLA_Breached (1 = Processing_Days > SLA_Days)",
    "metrics_tree": ev(pred),
    "metrics_baseline": ev(bpred),
    "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    "feature_importance": dict(sorted(agg.items(), key=lambda kv: -kv[1])),
    "breach_share_labeled": float(y.mean()),
}
joblib.dump(model, "trained_model.joblib")
with open("model_info.json", "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
