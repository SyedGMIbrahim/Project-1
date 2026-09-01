import pandas as pd
import numpy as np

def analyze():
    print("Loading dataset...")
    df = pd.read_csv("data/train/Diseases_and_Symptoms_dataset.csv")
    
    feature_cols = [col for col in df.columns if col != "diseases"]
    # Force convert to numeric in case there are string columns or spaces
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    symptom_counts = df[feature_cols].sum(axis=1)
    
    print("\nOverall Symptom Count Distribution:")
    print(f"Mean: {symptom_counts.mean():.2f}")
    print(f"Median: {symptom_counts.median():.2f}")
    print(f"Min: {symptom_counts.min():.2f}")
    print(f"Max: {symptom_counts.max():.2f}")
    print(f"Std Dev: {symptom_counts.std():.2f}")
    
    print("\nBreakdown by disease:")
    target_diseases = ["common cold", "seasonal allergies (hay fever)", "hyperemesis gravidarum"]
    
    for d in target_diseases:
        subset = df[df["diseases"] == d]
        counts = subset[feature_cols].sum(axis=1)
        print(f"\n--- {d} ---")
        print(f"Mean: {counts.mean():.2f}")
        print(f"Median: {counts.median():.2f}")
        print(f"Min: {counts.min():.2f}")
        print(f"Max: {counts.max():.2f}")
        print(f"Std Dev: {counts.std():.2f}")

if __name__ == "__main__":
    analyze()
