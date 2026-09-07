import asyncio
import numpy as np
from main import state, lifespan
from main import map_extracted_to_features
import ollama
import json

async def extract_symptoms_with_llm(text, feature_columns):
    prompt = f"""
    Extract all medical symptoms from the following text. Do not summarize. 
    Preserve anatomical and contextual details for generic symptoms (e.g., instead of "swelling", extract "swelling in big toe"; instead of "discharge", extract "crusty eye discharge").
    Respond ONLY with a valid JSON object containing a single key "symptoms" mapped to an array of strings.
    Example: {{"symptoms": ["runny nose", "congestion", "scratchy throat", "coughing", "sneezing", "body aches", "headache", "fever", "malaise"]}}
    
    Text: {text}
    """
    response = state.ollama_client.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": "You are a medical extractor. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        format="json",
    )
    parsed = json.loads(response["message"]["content"])
    extracted_list = parsed.get("symptoms", [])
    matched_symptoms = map_extracted_to_features(extracted_list, feature_columns)
    return matched_symptoms

async def run_tests():
    async with lifespan(None):
        feature_columns = state.classifier_features
        classifier = state.classifier
        
        tests = {
            "Cold": "I have a stuffy and runny nose, I'm sneezing a lot, coughing, and my throat is really scratchy and hurts.",
            "Migraine": "I have a terrible throbbing pain on one side of my head. I feel nauseous, and I can't stand bright lights or loud noises because they make it worse. I also saw some flickering zigzag lines before the headache started.",
            "Diabetes": "I've been feeling extremely thirsty lately and drinking tons of water, which means I'm peeing all the time too. I also noticed that a small cut on my foot is taking forever to heal.",
            "Chickenpox": "I have these itchy red spots that turned into little blisters all over my body. I also have a mild fever and feel super tired.",
            "UTI": "I feel a burning pain when I pee, I need to go to the bathroom constantly, and I can't hold it in sometimes. My lower stomach hurts too.",
            "Gout": "I woke up with sudden, severe pain in my right big toe. The joint is swollen, red, and feels hot to the touch.",
            "Conjunctivitis": "My eyes are very red and itchy. There is a yellowish crusty discharge, and it feels like there is grit in them. They are watering constantly.",
            "Hypothyroidism": "I'm always feeling cold, even when it's warm. I've gained weight without trying, and my hair is thinning and brittle."
        }
        
        for name, text in tests.items():
            print(f"\n--- Testing {name} ---")
            symptoms = await extract_symptoms_with_llm(text, feature_columns)
            print("Mapped symptoms:", symptoms)
            
            x = [1 if fname.lower() in symptoms else 0 for fname in feature_columns]
            arr = np.array(x).reshape(1, -1)
            
            proba = classifier.predict_proba(arr)[0]
            classes = list(classifier.classes_)
            sorted_predictions = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
            
            print("Top 3 Predictions:")
            for i in range(3):
                top_class, top_prob = sorted_predictions[i]
                print(f"  {i+1}. {top_class} ({top_prob:.4f})")

if __name__ == "__main__":
    asyncio.run(run_tests())
