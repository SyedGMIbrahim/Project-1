from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

LOGGER = logging.getLogger(__name__)


def train(csv_path: Path, output_dir: Path, test_size: float = 0.2, random_state: int = 42) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Support common variants: some datasets use 'prognosis' or 'diseases' as the target column
    if "disease" not in df.columns:
        if "prognosis" in df.columns:
            LOGGER.info("Renaming 'prognosis' column to 'disease'")
            df = df.rename(columns={"prognosis": "disease"})
        elif "diseases" in df.columns:
            LOGGER.info("Renaming 'diseases' column to 'disease'")
            df = df.rename(columns={"diseases": "disease"})

    if "disease" not in df.columns:
        raise ValueError("Expected a 'disease' column in the training CSV (or 'prognosis'/'diseases' to be present)")

    # Drop any trailing unnamed columns or columns that are entirely NaN
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    # never drop the target column even if it's weird
    cols_to_drop = sorted(set(unnamed_cols + all_nan_cols) - {"disease"})
    if cols_to_drop:
        LOGGER.info("Dropping columns that are unnamed or all-NaN: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    # Features are all columns except 'disease'
    feature_columns = [c for c in df.columns if c != "disease"]
    X = df[feature_columns]
    y = df["disease"]

    # Convert boolean/flag columns to numeric if necessary
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Apply SMOTE to balance rare disease classes in training data
    LOGGER.info("Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    LOGGER.info(
        "Training data resampled: original shape %s -> new shape %s",
        X_train.shape, X_train_resampled.shape
    )

    LOGGER.info("Building Ensemble Classifier...")
    # 1. Optimized Random Forest
    rf = RandomForestClassifier(
        n_estimators=150,
        criterion="entropy",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    
    # 2. ExtraTrees for variance reduction
    et = ExtraTreesClassifier(
        n_estimators=150,
        criterion="entropy",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    
    # 3. Histogram Gradient Boosting for error correction
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        random_state=random_state,
    )

    # 4. Combine them into a Soft Voting Classifier
    stacking_clf = StackingClassifier(
        estimators=[('rf', rf), ('et', et), ('hgb', hgb)],
        final_estimator=MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=random_state,
            early_stopping=True,
        ),
        cv=2,
        n_jobs=-1
    )

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', stacking_clf)
    ])

    LOGGER.info("Training ensemble model (this may take 30-60 seconds)...")
    clf.fit(X_train_resampled, y_train_resampled)

    # Evaluate on both train and test sets
    train_preds = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)

    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    LOGGER.info("="*60)
    LOGGER.info("Training Accuracy:   %.4f", train_acc)
    LOGGER.info("Validation Accuracy: %.4f", test_acc)
    LOGGER.info("="*60)
    LOGGER.info("\nDetailed Classification Report (Validation Set):\n%s", classification_report(
        y_test, test_preds, zero_division=0
    ))

    # Use test accuracy for threshold warnings
    acc = test_acc

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "symptom_classifier.joblib"
    # Save both model and feature list
    joblib.dump({"model": clf, "features": feature_columns}, model_path)

    LOGGER.info("Model saved to %s", model_path)

    if acc < 0.9:
        LOGGER.warning(
            "Model validation accuracy is below 90%% (%.4f); consider more data or tuning.", acc
        )
    else:
        LOGGER.info("SUCCESS: Validation accuracy exceeded 90%%!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a symptom->disease classifier and save it to disk.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to training CSV with symptoms and 'disease' column.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "models", help="Directory to write the trained model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    train(args.csv, args.out_dir, args.test_size, args.random_state)


if __name__ == "__main__":
    main()