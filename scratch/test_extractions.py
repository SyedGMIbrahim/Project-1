import asyncio
from main import extract_symptoms_with_llm, state, lifespan

async def run_tests():
    async with lifespan(None):
        feature_columns = state.classifier_features
        
        tests = {
            "Gout": "Sudden severe pain, redness, swelling, and heat in one big toe joint.",
            "Conjunctivitis": "Eye redness, itchiness, yellowish crusty discharge, gritty sensation, and watering.",
            "Hypothyroidism": "I've been feeling cold all the time, gained a lot of weight recently, and my hair is getting thin and brittle. Also dealing with fatigue and dry, rough skin."
        }
        
        for name, text in tests.items():
            print(f"\n--- Testing {name} ---")
            symptoms = extract_symptoms_with_llm(text, feature_columns)
            print("Mapped symptoms:", symptoms)

if __name__ == "__main__":
    asyncio.run(run_tests())
