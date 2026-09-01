import asyncio
from main import extract_symptoms_with_llm, state, lifespan, CLASSIFIER_PATH
import joblib

async def run_test():
    async with lifespan(None):
        text = "I've been feeling cold all the time, gained a lot of weight recently, and my hair is getting thin and brittle. Also dealing with fatigue and dry, rough skin."
        
        feature_columns = state.classifier_features
        print("Extracting symptoms for Hypothyroidism text...")
        symptoms = extract_symptoms_with_llm(text, feature_columns)
        print("Mapped symptoms:", symptoms)

if __name__ == "__main__":
    asyncio.run(run_test())
