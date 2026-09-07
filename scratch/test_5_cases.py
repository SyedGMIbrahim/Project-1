import joblib
import json
import requests
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/syedgmibrahim/Desktop/Project 1/backend")
PROD_MODEL_PATH = BASE_DIR / "models" / "symptom_classifier.joblib"
CALIB_MODEL_PATH = BASE_DIR / "models" / "symptom_classifier_calibrated.joblib"

cases = {
    "cold": "I have a runny nose, sore throat, and a slight cough.",
    "migraine": "Severe headache on one side of my head, feeling nauseous and the light hurts my eyes.",
    "diabetes": "I've been feeling extremely thirsty lately and have to pee all the time. Also feeling very tired.",
    "chickenpox": "My child has a fever, an itchy skin rash with blisters, and feels tired and irritable.",
    "UTI": "It burns when I urinate and I have to go constantly, but only a little comes out. My lower belly hurts."
}

def load_model(path):
    payload = joblib.load(path)
    if isinstance(payload, dict):
        return payload["model"], payload["features"]
    return payload, None

def get_predictions(model, features, symptoms):
    selected = set([s.lower() for s in symptoms])
    x = [1 if fname.lower() in selected else 0 for fname in features]
    arr = np.array(x).reshape(1, -1)
    
    if not hasattr(model, "predict_proba"):
        return [(model.predict(arr)[0], 1.0)]
        
    proba = model.predict_proba(arr)[0]
    classes = list(model.classes_)
    sorted_predictions = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
    return sorted_predictions[:3]

def run_test():
    print("Loading models...")
    prod_model, features = load_model(PROD_MODEL_PATH)
    calib_model, _ = load_model(CALIB_MODEL_PATH)
    
    for name, text in cases.items():
        print(f"\n{'='*60}\nTesting Case: {name.upper()}")
        print(f"Input: {text}")
        
        # 1. Extract Symptoms
        try:
            res = requests.post("http://127.0.0.1:8000/api/extract-symptoms", json={"text": text})
            res.raise_for_status()
            symptoms = res.json().get("symptoms", [])
            print(f"Mapped Symptoms: {symptoms}\n")
        except Exception as e:
            print(f"Extraction failed: {e}")
            continue
            
        if not symptoms:
            print("No symptoms mapped.")
            continue
            
        # 2. Diagnose with Production Model
        print("--- Production Model ---")
        prod_preds = get_predictions(prod_model, features, symptoms)
        print(f"Predicted Disease: {prod_preds[0][0]} (Confidence: {prod_preds[0][1]:.2%})")
        print("Top 3:")
        for d, p in prod_preds:
            print(f"  - {d}: {p:.2%}")
            
        # 3. Diagnose with Calibrated Model
        print("\n--- Calibrated Model ---")
        calib_preds = get_predictions(calib_model, features, symptoms)
        print(f"Predicted Disease: {calib_preds[0][0]} (Confidence: {calib_preds[0][1]:.2%})")
        print("Top 3:")
        for d, p in calib_preds:
            print(f"  - {d}: {p:.2%}")

if __name__ == "__main__":
    run_test()
