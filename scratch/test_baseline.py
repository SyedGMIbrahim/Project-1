import asyncio
import numpy as np
from main import extract_symptoms_with_llm, state, lifespan

async def run_tests():
    async with lifespan(None):
        feature_columns = state.classifier_features
        classifier = state.classifier
        
        tests = {
            "UTI": "I feel a burning pain when I pee, I need to go to the bathroom constantly, and I can't hold it in sometimes. My lower stomach hurts too.",
            "Common Cold": "I have a stuffy and runny nose, I'm sneezing a lot, coughing, and my throat is really scratchy and hurts.",
            "Chickenpox": "I have these itchy red spots that turned into little blisters all over my body. I also have a mild fever and feel super tired."
        }
        
        for name, text in tests.items():
            print(f"\n--- Testing {name} ---")
            symptoms = extract_symptoms_with_llm(text, feature_columns)
            print("Mapped symptoms:", symptoms)
            
            x = [1 if fname.lower() in symptoms else 0 for fname in feature_columns]
            arr = np.array(x).reshape(1, -1)
            
            proba = classifier.predict_proba(arr)[0]
            classes = list(classifier.classes_)
            sorted_predictions = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
            top_pred_class, top_pred_prob = sorted_predictions[0]
            
            print(f"Prediction: {top_pred_class} ({top_pred_prob:.4f})")

if __name__ == "__main__":
    asyncio.run(run_tests())
