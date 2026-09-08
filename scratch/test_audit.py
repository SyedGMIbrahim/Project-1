import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

features = ["frequent urination", "low urine output", "excessive anger", "irritability", "pelvic pain", "lower back pain"]
f_emb = np.array(emb.embed_documents(features))

syn = "irritability"
syn_emb = np.array(emb.embed_query(syn)).reshape(1, -1)
sims = cosine_similarity(syn_emb, f_emb)[0]

for f, s in zip(features, sims):
    print(f"{f}: {s:.4f}")
