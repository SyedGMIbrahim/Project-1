import json
import joblib
import ollama

def generate_synonyms():
    cls_payload = joblib.load("models/symptom_classifier.joblib")
    features = cls_payload["features"]
    
    client = ollama.Client(host="http://localhost:11434")
    
    synonyms_dict = {}
    
    chunk_size = 40
    for i in range(0, len(features), chunk_size):
        chunk = features[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}...")
        
        prompt = f"""
For each of the following medical symptoms, provide 3 to 5 common, lay-language phrases a patient might use to describe it.
Respond ONLY with a valid JSON object where the keys are the exact symptoms provided, and the values are arrays of strings (the synonyms).
Do not include any other text.
Symptoms:
{json.dumps(chunk)}
"""
        try:
            res = client.chat(
                model="llama3",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            parsed = json.loads(res["message"]["content"])
            # Ensure the keys match exactly
            for f in chunk:
                # the LLM might have changed casing
                for k, v in parsed.items():
                    if k.lower() == f.lower():
                        synonyms_dict[f] = [f.lower().replace("_", " ")] + [s.lower() for s in v if s.lower() != f.lower()]
                        break
                if f not in synonyms_dict:
                    synonyms_dict[f] = [f.lower().replace("_", " ")]
                    
        except Exception as e:
            print(f"Error processing chunk: {e}")
            for f in chunk:
                synonyms_dict[f] = [f.lower().replace("_", " ")]
    
    # Specific overrides requested by the user
    if "skin rash" in synonyms_dict:
        # Add requested overrides but keep it a set to remove duplicates, then list
        curr = set(synonyms_dict["skin rash"])
        curr.update(["skin rash", "blisters", "fluid-filled blisters", "red bumps", "itchy bumps"])
        synonyms_dict["skin rash"] = list(curr)
    
    with open("../scratch/generated_synonyms.json", "w") as f:
        json.dump(synonyms_dict, f, indent=4)
        
    print("Synonyms written to ../scratch/generated_synonyms.json")

if __name__ == "__main__":
    generate_synonyms()
