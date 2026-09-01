import json
from main import map_extracted_to_features, state
import joblib
cls = joblib.load("./models/symptom_classifier.joblib")
features = list(cls["features"])

extracted = ["runny or stuffy nose", "congestion", "sore or scratchy throat", "coughing", "sneezing", "body aches", "mild headache", "low-grade fever", "malaise"]

matched = map_extracted_to_features(extracted, features)
print("Matched:", matched)
