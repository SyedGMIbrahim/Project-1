import joblib
import pandas as pd
import numpy as np

# Load model
print("Loading model...")
model_data = joblib.load("backend/models/symptom_classifier_augmented.joblib")
clf = model_data["model"]
features = model_data["features"]

scaler = clf.named_steps["scaler"]
stacking_clf = clf.named_steps["classifier"]

print(f"StackingClassifier passthrough: {stacking_clf.passthrough}")

# Create test data (Diabetes case: 'frequent urination', 'fatigue')
input_data = pd.DataFrame(0, index=[0], columns=features)
for symptom in ['frequent urination', 'fatigue']:
    if symptom in features:
        input_data.at[0, symptom] = 1

# Check 1: StandardScaler transform
print("\n--- Checking StandardScaler ---")
X_scaled = scaler.transform(input_data)
print(f"NaN in X_scaled: {np.isnan(X_scaled).any()}")
print(f"Inf in X_scaled: {np.isinf(X_scaled).any()}")
print(f"Max value in X_scaled: {np.max(X_scaled)}")
print(f"Min value in X_scaled: {np.min(X_scaled)}")

# Check 2: Base estimators output
print("\n--- Checking Base Estimators ---")
base_preds = []
for i, estimator in enumerate(stacking_clf.estimators_):
    preds = estimator.predict_proba(X_scaled)
    print(f"Estimator {i} predictions - NaN: {np.isnan(preds).any()}, Inf: {np.isinf(preds).any()}")
    base_preds.append(preds)

if hasattr(stacking_clf, "estimators_"):
    # Stacking classifier concatenates the predictions
    stacked_X = np.concatenate(base_preds, axis=1)
    print(f"\nStacked X shape: {stacked_X.shape}")
    print(f"NaN in Stacked X: {np.isnan(stacked_X).any()}")
    print(f"Inf in Stacked X: {np.isinf(stacked_X).any()}")

    # Check 3: Meta-learner output
    print("\n--- Checking Meta-Learner (MLP) ---")
    meta_preds = stacking_clf.final_estimator_.predict_proba(stacked_X)
    print(f"Meta-learner predictions - NaN: {np.isnan(meta_preds).any()}, Inf: {np.isinf(meta_preds).any()}")
    print(f"Sum of meta_preds: {np.sum(meta_preds, axis=1)}")

print("\n--- Checking Feature / Label alignment ---")
import sys
sys.path.append('backend')
from train_classifier import augment_data

df = pd.read_csv("backend/data/train/Diseases_and_Symptoms_dataset.csv")
if "disease" not in df.columns:
    if "prognosis" in df.columns:
        df = df.rename(columns={"prognosis": "disease"})
    elif "diseases" in df.columns:
        df = df.rename(columns={"diseases": "disease"})
X = df.drop(columns=["disease"])
y = df["disease"]
X = X.fillna(0)

print(f"Original X shape: {X.shape}, y shape: {y.shape}")
print(f"Features trained on match original data features: {list(X.columns) == features}")

# Test augmentation alignment
X_aug, y_aug = augment_data(X.head(5), y.head(5), random_state=42)
print("First row of augmented labels vs original labels:")
for i in range(5):
    print(f"Row {i} original label: {y.iloc[i]}, augmented label (row {i+5}): {y_aug.iloc[i+5]}")
