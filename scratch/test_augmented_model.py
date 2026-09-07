import requests
import json
import time
import joblib
import pandas as pd
import numpy as np

# Load the new augmented model
model_data = joblib.load("backend/models/symptom_classifier_augmented.joblib")
clf = model_data["model"]
features = model_data["features"]
classes = clf.classes_

def test_case(name, text):
    print(f"\n--- Testing {name} ---")
    start = time.time()
    resp = requests.post("http://127.0.0.1:8000/api/extract-symptoms", json={"text": text})
    extract_time = time.time() - start
    
    if resp.status_code != 200:
        print("Extraction failed:", resp.text)
        return
        
    mapped_symptoms = resp.json().get("symptoms", [])
    print(f"Extraction Latency: {extract_time:.3f}s")
    print(f"Mapped Symptoms: {mapped_symptoms}")
    
    if mapped_symptoms:
        # Replicate main.py diagnosis logic with the new model
        input_data = pd.DataFrame(0, index=[0], columns=features)
        
        for symptom in mapped_symptoms:
            if symptom in features:
                input_data.at[0, symptom] = 1
                
        # Drop categorical/non-feature columns if they somehow got in
        input_data = input_data[features] 
        
        # Predict
        probas = clf.predict_proba(input_data)[0]
        max_idx = np.argmax(probas)
        predicted = classes[max_idx]
        probability = probas[max_idx]
        
        print(f"Prediction: {predicted} ({probability:.2%})")
    else:
        print("No symptoms mapped, skipping diagnosis.")

cold_text = "Runny or stuffy nose and congestion. A sore or scratchy throat, coughing, and sneezing. Slight body aches, a mild headache, and sometimes a low-grade fever. A general feeling of being unwell (malaise)."
migraine_text = "throbbing, pulsing pain on one side of my head... bright lights and loud noises make it worse... nauseous... flickering zigzag lines in my vision"
diabetes_text = "excessive thirst, frequent urination, unintentional weight loss, fatigue, and slow-healing wounds"
chickenpox_text = "rash of itchy red bumps progressing to fluid-filled blisters and crusting, plus fever and irritability"
uti_text = "strong, persistent urge to urinate, a burning sensation when urinating, passing frequent, small amounts of urine, and cloudy urine."

test_case("Common Cold", cold_text)
test_case("Migraine", migraine_text)
test_case("Diabetes", diabetes_text)
test_case("Chickenpox", chickenpox_text)
test_case("UTI", uti_text)
