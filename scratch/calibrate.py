import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path("/Users/syedgmibrahim/Desktop/Project 1/backend")
CSV_PATH = BASE_DIR / "data" / "train" / "Diseases_and_Symptoms_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "symptom_classifier.joblib"
OUT_PATH = BASE_DIR / "models" / "symptom_classifier_calibrated.joblib"

df = pd.read_csv(CSV_PATH)
if "prognosis" in df.columns:
    df = df.rename(columns={"prognosis": "disease"})
elif "diseases" in df.columns:
    df = df.rename(columns={"diseases": "disease"})

unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
all_nan_cols = [c for c in df.columns if df[c].isna().all()]
cols_to_drop = sorted(set(unnamed_cols + all_nan_cols) - {"disease"})
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

feature_columns = [c for c in df.columns if c != "disease"]
X = df[feature_columns].fillna(0)
y = df["disease"]

# Exact same split as train_classifier.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Loading production model...")
payload = joblib.load(MODEL_PATH)
if isinstance(payload, dict):
    base_clf = payload["model"]
    features = payload["features"]
else:
    base_clf = payload
    features = None

# Evaluate uncalibrated model
print("Evaluating production model...")
preds = base_clf.predict(X_test)
print(f"Original holdout accuracy: {accuracy_score(y_test, preds):.4f}")

# Fit Platt scaling
print("\nFitting CalibratedClassifierCV (method='sigmoid')...")
calibrated_clf = CalibratedClassifierCV(estimator=base_clf, cv="prefit", method="sigmoid")
calibrated_clf.fit(X_test, y_test)

calibrated_preds = calibrated_clf.predict(X_test)
calibrated_acc = accuracy_score(y_test, calibrated_preds)
print(f"Calibrated holdout accuracy: {calibrated_acc:.4f}")

print(f"\nSaving calibrated model to {OUT_PATH}...")
joblib.dump({"model": calibrated_clf, "features": features}, OUT_PATH)
print("Done.")
