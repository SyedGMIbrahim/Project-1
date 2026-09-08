import requests
import json
import time

URL_EXTRACT = "http://localhost:8000/api/extract-symptoms"
URL_PREDICT = "http://localhost:8000/api/diagnose"

CASES = {
    "Cold": "I've been feeling terrible for the last two days. I have a high fever and my whole body aches. My throat is incredibly sore and scratchy. My nose won't stop running, and I have a persistent cough that keeps me up at night.",
    "Migraine": "I have this intense, throbbing headache on the right side of my head. It's so bad that it's making me feel nauseous and sick to my stomach.",
    "Diabetes": "Lately I've been experiencing extreme thirst, no matter how much water I drink. I have to go to the bathroom for frequent urination all the time. I've also noticed some blurred vision and this strange tingling in my feet.",
    "Chickenpox": "My son woke up with a high fever and a bad headache. Today, we noticed a red, intensely itchy skin rash starting to form on his stomach and back.",
    "UTI": "I'm having a lot of painful urination and my urine volume is very low even though I feel the urge for frequent urination constantly. I also noticed an unusual color or odor to my urine and have lower abdominal pain.",
    "Gout": "I woke up in the middle of the night with sudden, severe pain in my big toe. The joint is red, swollen, and hot to the touch, and it hurts so much that even the weight of a bedsheet on it is unbearable.",
    "Conjunctivitis": "My right eye has been red and itchy for the past two days, and there's a thick, yellowish discharge that crusts over my eyelashes, especially when I wake up in the morning. My eye also feels gritty, like there's sand in it, and it's been watering a lot.",
    "Hypothyroidism": "I've been having sudden chills recently. I also noticed some unexpected weight gain even though my diet hasn't changed. I'm constantly fighting sleepiness during the day, and my skin feels very dry.",
    "Sinusitis": "My nose has been producing thick, greenish-yellow nasal discharge for over a week. I have a lot of pressure and pain around my cheeks and forehead. I have also lost my sense of smell almost completely, and have a bad taste in my mouth. I also have a mild fever.",
    "Kidney Stones": "I have been experiencing sudden, severe cramping pain in my lower back and side that comes in waves, radiating down toward my groin. My urine has also been an unusual color, and I have been feeling nauseous and have vomited once.",
    "Vaginal Infection": "I've been experiencing a lot of itching and burning around my vaginal area, along with a thick, white vaginal discharge. It also hurts a bit when I pee.",
    "Asthma": "I've been having a lot of trouble breathing lately, feeling a tightness in my chest. When I try to take a deep breath, I start wheezing and coughing, especially at night.",
    "Shingles": "A few days ago I had a tingling and burning pain on the left side of my torso. Now there is a painful, blistering red rash wrapped around that side of my body. I also have a mild headache."
}

def run_stage1_tests():
    results = []
    print("Running Stage 1: Three-State Symptom Mapping Tests\n")
    
    for name, text in CASES.items():
        print(f"\n--- Testing {name} ---")
        try:
            # 1. Extract symptoms
            ext_res = requests.post(URL_EXTRACT, json={"description": text})
            if ext_res.status_code != 200:
                print(f"Error extracting for {name}: {ext_res.text}")
                results.append((name, [], [], [], "ERROR", 0))
                continue
            
            ext_data = ext_res.json()
            supported = ext_data.get("symptoms", [])
            uncertain = ext_data.get("uncertain", [])
            unsupported = ext_data.get("unsupported", [])
            
            print(f"  SUPPORTED ({len(supported)}): {supported}")
            if uncertain:
                print(f"  UNCERTAIN ({len(uncertain)}): {uncertain}")
            else:
                print("  UNCERTAIN (0): []")
            print(f"  UNSUPPORTED ({len(unsupported)}): {unsupported}")
            
            if not supported:
                results.append((name, supported, uncertain, unsupported, "Unknown", 0))
                continue
                
            # 2. Predict using supported symptoms only
            pred_res = requests.post(URL_PREDICT, json={"symptoms": supported})
            if pred_res.status_code != 200:
                results.append((name, supported, uncertain, unsupported, "ERROR", 0))
                continue
                
            pred_data = pred_res.json()
            prediction = pred_data.get("predicted", "Unknown")
            probability = pred_data.get("probability", 0)
            
            print(f"  PREDICTION: {prediction} ({probability*100:.1f}%)")
            
            results.append((name, supported, uncertain, unsupported, prediction, probability))
            
        except Exception as e:
            print(f"Exception on {name}: {e}")
            results.append((name, [], [], [], "EXCEPTION", 0))
            
        time.sleep(0.5)

    print("\n\n### Stage 1: 13-Case Regression Results Summary\n")
    print("| Case | Supported | Uncertain | Unsupported | Prediction | Confidence |")
    print("|---|---|---|---|---|---|")
    for name, supp, unc, unsupp, pred, prob in results:
        supp_str = str(len(supp))
        unc_str = str(len(unc))
        unsupp_str = str(len(unsupp))
        print(f"| **{name}** | {supp_str} | {unc_str} | {unsupp_str} | {pred} | {prob*100:.1f}% |")

if __name__ == "__main__":
    run_stage1_tests()
