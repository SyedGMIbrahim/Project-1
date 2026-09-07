import joblib
cls = joblib.load("./models/symptom_classifier.joblib")
features = [f.lower().replace("_", " ") for f in cls["features"]]

keywords = ["light", "photo", "sound", "noise", "aura", "vision", "head", "eye", "migraine"]
print("Searching for keywords:", keywords)
for f in features:
    for k in keywords:
        if k in f:
            print(f"Match for '{k}': {f}")
