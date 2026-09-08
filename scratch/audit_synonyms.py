import json
import joblib
import numpy as np
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "../backend")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
SYNONYMS_FILE = os.path.join(BACKEND_DIR, "approved_synonyms_v1.json")

def main():
    import pandas as pd
    dataset_path = os.path.join(BACKEND_DIR, "data", "train", "Diseases_and_Symptoms_dataset.csv")
    df = pd.read_csv(dataset_path, nrows=0)
    features = [c for c in df.columns if c not in ("Disease", "ID", "Unnamed: 0", "index")]
    feature_lower = [f.lower().replace("_", " ") for f in features]
    
    with open(SYNONYMS_FILE, "r") as f:
        generated_synonyms = json.load(f)
        
    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={"device": "cpu"})
    
    # Pre-embed all base features
    feature_embeddings = np.array(emb_model.embed_documents(feature_lower))
    
    flagged = []
    
    for assigned_feature, synonyms in generated_synonyms.items():
        if assigned_feature not in feature_lower:
            print(f"Warning: Assigned feature '{assigned_feature}' not in schema.")
            continue
            
        assigned_idx = feature_lower.index(assigned_feature)
        
        for syn in synonyms:
            syn_emb = np.array(emb_model.embed_query(syn)).reshape(1, -1)
            sims = cosine_similarity(syn_emb, feature_embeddings)[0]
            
            top_indices = sims.argsort()[-5:][::-1]
            top_1_idx = top_indices[0]
            
            if top_1_idx != assigned_idx:
                # The synonym is closer to another column than its assigned column
                score_assigned = sims[assigned_idx]
                score_top1 = sims[top_1_idx]
                top1_feature = feature_lower[top_1_idx]
                
                # Flag if the difference is meaningful (e.g., top1 > assigned by 0.02)
                # or if top1 is very high and assigned is lower
                if score_top1 > score_assigned + 0.01:
                    flagged.append({
                        "synonym": syn,
                        "assigned_feature": assigned_feature,
                        "assigned_score": float(score_assigned),
                        "top_1_feature": top1_feature,
                        "top_1_score": float(score_top1),
                        "delta": float(score_top1 - score_assigned)
                    })

    # Sort flagged by delta (largest first)
    flagged.sort(key=lambda x: x["delta"], reverse=True)
    
    print(f"--- Synonym Audit Results (Total flagged: {len(flagged)}) ---")
    for item in flagged:
        print(f"Synonym: '{item['synonym']}'")
        print(f"  Assigned: [{item['assigned_feature']}] (Score: {item['assigned_score']:.4f})")
        print(f"  Closer to: [{item['top_1_feature']}] (Score: {item['top_1_score']:.4f})")
        print(f"  Delta: +{item['delta']:.4f}\n")

if __name__ == "__main__":
    main()
