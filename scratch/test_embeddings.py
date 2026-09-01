import json
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load features
cls = joblib.load("./models/symptom_classifier.joblib")
features = [f.lower().replace("_", " ") for f in cls["features"]]

print("Loading embeddings...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding features...")
feature_embeddings = embeddings_model.embed_documents(features)
feature_embeddings = np.array(feature_embeddings)

test_phrases = [
    "throbbing, pulsing pain on one side of my head",
    "bright lights and loud noises make it worse",
    "flickering zigzag lines in my vision",
    "nauseous"
]

print("\nTesting Migraine Phrases:")
for phrase in test_phrases:
    phrase_emb = np.array(embeddings_model.embed_query(phrase)).reshape(1, -1)
    sims = cosine_similarity(phrase_emb, feature_embeddings)[0]
    
    top_indices = sims.argsort()[-3:][::-1]
    print(f"\nPhrase: '{phrase}'")
    for i in top_indices:
        print(f"  - {features[i]} (score: {sims[i]:.3f})")
