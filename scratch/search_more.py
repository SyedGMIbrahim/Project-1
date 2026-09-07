import joblib
cls = joblib.load("./models/symptom_classifier.joblib")
features = [f.lower().replace("_", " ") for f in cls["features"]]

keywords = ["ache", "ill", "coryza"]
for f in features:
    for k in keywords:
        if k in f:
            print(f"Match for '{k}': {f}")
