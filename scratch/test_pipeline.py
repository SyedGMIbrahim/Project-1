import requests
import json
import time

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
        diag_resp = requests.post("http://127.0.0.1:8000/api/diagnose", json={"symptoms": mapped_symptoms})
        if diag_resp.status_code == 200:
            diag = diag_resp.json()
            print(f"Prediction: {diag['predicted']} ({diag['probability']:.2%})")
        else:
            print("Diagnosis failed:", diag_resp.text)
    else:
        print("No symptoms mapped, skipping diagnosis.")

cold_text = "Runny or stuffy nose and congestion. A sore or scratchy throat, coughing, and sneezing. Slight body aches, a mild headache, and sometimes a low-grade fever. A general feeling of being unwell (malaise)."
migraine_text = "throbbing, pulsing pain on one side of my head... bright lights and loud noises make it worse... nauseous... flickering zigzag lines in my vision"
diabetes_text = "excessive thirst, frequent urination, unintentional weight loss, fatigue, and slow-healing wounds"
chickenpox_text = "rash of itchy red bumps progressing to fluid-filled blisters and crusting, plus fever and irritability"

test_case("Common Cold", cold_text)
test_case("Migraine", migraine_text)
test_case("Diabetes", diabetes_text)
test_case("Chickenpox", chickenpox_text)
