import asyncio
from main import diagnose, DiagnoseRequest, state, lifespan

async def run_tests():
    async with lifespan(None):
        print("Testing UTI (Expected: Abstention)")
        # UTI mapped symptoms
        req = DiagnoseRequest(symptoms=['involuntary urination', 'painful urination', 'frequent urination', 'lower abdominal pain'])
        res = diagnose(req)
        print("Result:", res)
        
        print("\nTesting Conjunctivitis (Expected: Confident)")
        # Conjunctivitis mapped symptoms
        req = DiagnoseRequest(symptoms=['white discharge from eye', 'itchiness of eye', 'eye redness', 'lacrimation'])
        res = diagnose(req)
        print("Result:", res)

if __name__ == "__main__":
    asyncio.run(run_tests())
