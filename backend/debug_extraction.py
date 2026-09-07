import difflib
import json
from pathlib import Path
import joblib
import ollama

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "symptom_classifier.joblib"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

def run_debug():
    try:
        cls_payload = joblib.load(CLASSIFIER_PATH)
        feature_columns = list(cls_payload["features"])
    except Exception as e:
        print(f"Failed to load classifier features: {e}")
        return

    client = ollama.Client(host=DEFAULT_OLLAMA_HOST)

    text = "Runny or stuffy nose and congestion. A sore or scratchy throat, coughing, and sneezing. Slight body aches, a mild headache, and sometimes a low-grade fever. A general feeling of being unwell (malaise)."

    prompt = f"""
    Extract all medical symptoms from the following text. Do not summarize. 
    Respond ONLY with a valid JSON object containing a single key "symptoms" mapped to an array of strings.
    Example: {{"symptoms": ["runny nose", "congestion", "scratchy throat", "coughing", "sneezing", "body aches", "headache", "fever", "malaise"]}}
    
    Text: {text}
    """

    print("Querying LLM...\n")
    response = client.chat(
        model=DEFAULT_OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a medical extractor. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        format="json",
    )

    content = response["message"]["content"]
    print("=== RAW LLM JSON OUTPUT ===")
    print(content)
    print("===========================\n")

    try:
        parsed = json.loads(content)
        extracted_list = parsed.get("symptoms", [])
    except json.JSONDecodeError:
        print("Failed to parse JSON")
        return

    feature_lower = {f: f.lower().replace("_", " ") for f in feature_columns}
    final_symptoms = set()

    print("=== DIFFLIB MATCHING DEBUG ===")
    for symptom in extracted_list:
        symptom_clean = str(symptom).lower().strip()
        print(f"Extracted string: '{symptom_clean}'")
        if len(symptom_clean) < 4:
            print(f"  -> Skipped (length < 4)\n")
            continue
        
        # Get top 3 candidates and their scores
        scored_candidates = []
        for orig, lowered in feature_lower.items():
            score = difflib.SequenceMatcher(None, symptom_clean, lowered).ratio()
            scored_candidates.append((orig, lowered, score))
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        print(f"  Top 3 candidates:")
        for i in range(3):
            if i < len(scored_candidates):
                print(f"    - {scored_candidates[i][1]} (score: {scored_candidates[i][2]:.4f}) -> mapped column: '{scored_candidates[i][0]}'")
                
        # The actual mapping logic from main.py
        matches = difflib.get_close_matches(symptom_clean, feature_lower.values(), n=1, cutoff=0.85)
        if matches:
            for orig, lowered in feature_lower.items():
                if lowered == matches[0]:
                    final_symptoms.add(orig)
                    print(f"  -> MATCHED VIA FUZZY: '{orig}' (Score: {difflib.SequenceMatcher(None, symptom_clean, lowered).ratio():.4f})")
                    break
        else:
            matched_exact = False
            for orig, lowered in feature_lower.items():
                # Exact word matching logic
                if f" {symptom_clean} " in f" {lowered} " or symptom_clean == lowered:
                    final_symptoms.add(orig)
                    print(f"  -> MATCHED VIA EXACT SUBSTRING: '{orig}'")
                    matched_exact = True
            if not matched_exact:
                print(f"  -> NO MATCH")
        print()
        
    print("=== FINAL FEATURE VECTOR (Set to 1) ===")
    if not final_symptoms:
        print("None")
    for f in sorted(list(final_symptoms)):
        print(f"- {f}")
    print("=======================================\n")

if __name__ == "__main__":
    run_debug()
