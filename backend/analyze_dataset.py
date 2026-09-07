import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def run_analysis():
    df = pd.read_csv('data/train/Diseases_and_Symptoms_dataset.csv')
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
    
    # Check actual class names
    y = y.str.lower().str.strip()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Total SMOTE resampled rows: {len(X_train_resampled)}")

    sums = X_train_resampled.sum(axis=1)
    stats = {
        'mean': sums.mean(),
        'median': sums.median(),
        'min': sums.min(),
        'max': sums.max(),
        'std': sums.std()
    }
    print(f"Overall Stats: {stats}")

    print(f"\n--- Original Dataset Analysis ---")
    orig_sums = X.sum(axis=1)
    d_mask_orig = y == "hyperemesis gravidarum"
    d_sums_orig = orig_sums[d_mask_orig]
    print(f"Original HG rows: {len(d_sums_orig)}")
    if len(d_sums_orig) > 0:
        print(f"Original HG Stats: mean={d_sums_orig.mean():.2f}, median={d_sums_orig.median()}, min={d_sums_orig.min()}, max={d_sums_orig.max()}")
    else:
        print("No original HG rows found!")

    print(f"\n--- SMOTE Dataset Analysis ---")
    d_mask_smote = y_train_resampled == "hyperemesis gravidarum"
    d_sums_smote = sums[d_mask_smote]
    print(f"SMOTE HG rows: {len(d_sums_smote)}")
    print(f"SMOTE HG Stats: mean={d_sums_smote.mean():.2f}, median={d_sums_smote.median()}, min={d_sums_smote.min()}, max={d_sums_smote.max()}")

if __name__ == "__main__":
    run_analysis()
