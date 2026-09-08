import json
import numpy as np
from main import state, app
from contextlib import asynccontextmanager
import asyncio

async def main():
    async with app.router.lifespan_context(app):
        import time
        time.sleep(2)

        def print_top(phrase):
            print(f"\n--- {phrase} ---")
            emb = state.embeddings_model.encode(phrase)
            sims = np.dot(state.feature_embeddings, emb) / (
                np.linalg.norm(state.feature_embeddings, axis=1) * np.linalg.norm(emb)
            )
            top_idx = np.argsort(sims)[::-1][:5]
            for idx in top_idx:
                print(f"{state.embedding_to_feature[idx]}: {sims[idx]:.3f}")

        print_top("loss of sense of smell")
        print_top("pain that is unbearable")

if __name__ == "__main__":
    asyncio.run(main())
import time
time.sleep(2)

def print_top(phrase):
    print(f"\n--- {phrase} ---")
    emb = state.embeddings_model.encode(phrase)
    sims = np.dot(state.feature_embeddings, emb) / (
        np.linalg.norm(state.feature_embeddings, axis=1) * np.linalg.norm(emb)
    )
    top_idx = np.argsort(sims)[::-1][:5]
    for idx in top_idx:
        print(f"{state.embedding_to_feature[idx]}: {sims[idx]:.3f}")

print_top("loss of sense of smell")
print_top("pain that is unbearable")
