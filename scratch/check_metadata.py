import asyncio
import os
import sys
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

def check_mtsamples_duplicates():
    print("===========================================")
    print("CHECKING MTSAMPLES.CSV FOR DUPLICATES")
    print("===========================================")
    csv_path = Path("backend/data/mentor_dataset/mtsamples.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Total rows in mtsamples.csv: {len(df)}")
    
    # check for exact duplicates in the text column
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("transcription") or cols.get("text")
    if not text_col:
        print("Could not find text column")
        return
        
    dupes = df.duplicated(subset=[text_col], keep=False)
    num_dupes = dupes.sum()
    print(f"Total rows with duplicate transcriptions: {num_dupes}")
    if num_dupes > 0:
        print("\nExample duplicates:")
        sample_dupes = df[dupes].sort_values(by=[text_col]).head(6)
        for _, row in sample_dupes.iterrows():
            text = str(row[text_col])[:100].replace('\n', ' ')
            print(f"- ID: {row.get('Unnamed: 0', 'N/A')} | Spec: {row.get('medical_specialty', 'N/A')[:20]} | Text: {text}...")

def check_metadata():
    from main import state, lifespan
    
    print("\n===========================================")
    print("CHECKING CHROMADB METADATA")
    print("===========================================")
    
    async def run():
        async with lifespan(None):
            if state.vector_store is None:
                print("Vector store not initialized.")
                return
                
            # Let's retrieve everything that matches the sinusitis report
            # We can use similarity search or just direct get with where clause
            collection = state.vector_store._collection
            
            # 1. Check if we have chunks with file_name = Sinusitis_Sample_Medical_Report-4884886573.pdf
            pdf_name = "Sinusitis_Sample_Medical_Report-4884886573.pdf"
            print(f"Querying Chroma for metadata where file_name == '{pdf_name}'")
            try:
                results = collection.get(where={"file_name": pdf_name})
                metadatas = results.get("metadatas", [])
                print(f"Found {len(metadatas)} chunks for {pdf_name}")
                if metadatas:
                    print("Sample metadata from first chunk:")
                    print(metadatas[0])
            except Exception as e:
                print(f"Error querying Chroma: {e}")

            # 2. Check the duplicates for the neurological exam snippet
            print("\nQuerying Chroma for the duplicate neurological chunks...")
            res = state.vector_store.similarity_search("What is the patient's diagnosis?", k=3)
            for i, doc in enumerate(res):
                print(f"\nResult {i+1} Metadata:")
                print(doc.metadata)

    asyncio.run(run())

if __name__ == "__main__":
    check_mtsamples_duplicates()
    check_metadata()
