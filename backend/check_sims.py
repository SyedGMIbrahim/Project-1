import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

def run():
    print("Loading classifier...")
    cls_payload = joblib.load("models/symptom_classifier.joblib")
    features = cls_payload["features"]
    
    print(f"Loaded {len(features)} features.")
    print("Printing all 230 features for review:")
    print(", ".join(features))
    
    print("\nLoading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    test_phrases = ["fluid-filled blisters", "itchy red bumps"]
    target_columns = ["skin rash", "skin lesion"]
    
    # Just a sanity check for alternate phrasing for diabetes
    alt_phrases = ["polydipsia", "polyuria", "increased thirst", "dehydration", "unhealed sores", "delayed healing", "cachexia", "polyphagia"]
    
    print("\n--- Cosine Similarities against skin rash / skin lesion ---")
    test_embs = embeddings.embed_documents(test_phrases)
    target_embs = embeddings.embed_documents(target_columns)
    
    for i, t_p in enumerate(test_phrases):
        for j, t_c in enumerate(target_columns):
            sim = cosine_similarity(np.array(test_embs[i]).reshape(1, -1), np.array(target_embs[j]).reshape(1, -1))[0][0]
            print(f"'{t_p}' vs '{t_c}': {sim:.3f}")

    print("\n--- Testing alternate diabetes phrasings against all 230 features ---")
    feature_embs = np.array(embeddings.embed_documents(features))
    alt_embs = np.array(embeddings.embed_documents(alt_phrases))
    
    for i, a_p in enumerate(alt_phrases):
        sims = cosine_similarity(alt_embs[i].reshape(1, -1), feature_embs)[0]
        top_indices = sims.argsort()[-3:][::-1]
        print(f"\n'{a_p}':")
        for idx in top_indices:
            print(f"  -> '{features[idx]}' (score: {sims[idx]:.3f})")

if __name__ == '__main__':
    run()
