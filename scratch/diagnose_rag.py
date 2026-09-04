import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from ingest import read_pdf_file, build_text_splitter, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from langchain_core.documents import Document

def main():
    pdf_path = Path("backend/data/raw/Sinusitis_Sample_Medical_Report-4884886573.pdf")
    
    print("===========================================")
    print("1. PDF PARSING CHECK")
    print("===========================================")
    print(f"File: {pdf_path.name}")
    try:
        import pdfplumber
        print("pdfplumber is available in the environment.")
    except ImportError:
        print("pdfplumber is NOT available in the environment. Using pypdf fallback.")
        
    page_texts = read_pdf_file(pdf_path)
    print(f"\nNumber of pages extracted: {len(page_texts)}")
    for i, (page_num, text) in enumerate(page_texts):
        print(f"\n--- PAGE {page_num} RAW TEXT ---")
        print(text)
        print("------------------------------")
        
    print("\n===========================================")
    print("2. CHUNKING CHECK")
    print("===========================================")
    print(f"Config: chunk_size={DEFAULT_CHUNK_SIZE}, chunk_overlap={DEFAULT_CHUNK_OVERLAP}")
    splitter = build_text_splitter(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    
    source_documents = []
    for page_number, text in page_texts:
        source_documents.append(Document(page_content=text, metadata={"page": page_number}))
        
    chunked_documents = splitter.split_documents(source_documents)
    print(f"Total chunks generated: {len(chunked_documents)}")
    from main import state, lifespan, build_prompt
    
    print("\n===========================================")
    print("3. RETRIEVAL CHECK")
    print("===========================================")
    test_query = "What is the patient's diagnosis?"
    print(f"Query: '{test_query}'")
    
    async def run_retrieval_and_gen():
        async with lifespan(None):
            if state.vector_store is None:
                print("Vector store not initialized.")
                return
            
            retrieved_documents = state.vector_store.similarity_search(test_query, k=3)
            print(f"\nRetrieved {len(retrieved_documents)} chunks.")
            for i, doc in enumerate(retrieved_documents):
                print(f"\n--- RETRIEVED CHUNK {i+1} ---")
                print(f"Source: {doc.metadata.get('file_name', 'Unknown')}")
                print(f"Content: {doc.page_content}")
                print("-------------------------")
                
            print("\n===========================================")
            print("4. GENERATION CHECK")
            print("===========================================")
            context_chunks = [document.page_content for document in retrieved_documents]
            prompt = build_prompt(test_query, context_chunks)
            print("--- EXACT PROMPT SENT TO LLM ---")
            print(prompt)
            print("--------------------------------")
            
            try:
                print("\nCalling LLM...")
                ollama_response = state.ollama_client.chat(
                    model="llama3",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert clinical AI. Using the retrieved context, provide a highly "
                                "detailed, comprehensive, and multi-paragraph answer to the user's question. "
                                "Explain the reasoning clearly."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0},
                )
                print("\n--- LLM RESPONSE ---")
                print(ollama_response["message"]["content"].strip())
                print("--------------------")
            except Exception as e:
                print(f"LLM Generation failed: {e}")

    asyncio.run(run_retrieval_and_gen())

if __name__ == "__main__":
    main()
