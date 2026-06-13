"""
RAG (Retrieval-Augmented Generation) Chatbot for Skin Disease Q&A.
Loads medical PDFs, builds a FAISS index, and answers questions with citations.
Supports English and Roman Urdu answers.

Usage:
    python -m src.rag --build_index
    python -m src.rag --ask "What are common warning signs of melanoma?"
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Medical Disclaimer
DISCLAIMER = (
    "\n[MEDICAL DISCLAIMER]: This assistant provides educational information only. "
    "It is not a medical diagnosis. For urgent symptoms or suspicious lesions, "
    "please consult a qualified dermatologist or doctor immediately."
)

# Roman Urdu Detection Keywords
URDU_KEYWORDS = ["kya", "kaise", "kyun", "ilaaj", "alamat", "bimari", "hai", "hain", "ka", "ki", "ke", "ko"]

def detect_language(text, requested_lang="auto"):
    """Roughly detect if text is English or Roman Urdu."""
    if requested_lang != "auto":
        return requested_lang
    
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)
    urdu_count = sum(1 for w in words if w in URDU_KEYWORDS)
    
    if urdu_count >= 1:
        return "roman_urdu"
    return "english"

def translate_to_roman_urdu(text):
    """
    Very basic rule-based conversion for specific phrases.
    In a real scenario, this would use a translation LLM.
    Since we avoid paid APIs/heavy local LLMs, we provide a functional extractive answer.
    """
    replacements = {
        "Source": "Zariya",
        "Page": "Safha",
        "I could not find enough relevant information": "Mujhe faraham karda medical documents mein iska sahi jawab nahi mila",
        "Based on the documents": "Documents ke mutabiq",
    }
    for eng, urd in replacements.items():
        text = text.replace(eng, urd)
    return text

def build_index(docs_dir, index_dir):
    """Load PDFs, chunk text, and save FAISS index."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"[ERROR] Documents directory not found: {docs_dir}")
        return

    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARNING] No PDF files found in {docs_dir}")
        return

    all_docs = []
    print(f"Loading {len(pdf_files)} PDFs...")
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"  Failed to load {pdf.name}: {e}")

    if not all_docs:
        print("[ERROR] No text extracted from PDFs.")
        return

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} text chunks.")

    # Embeddings
    print("Generating embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    print(f"FAISS index saved to {index_dir}")

def ask_question(query, index_dir, top_k=4, language="auto"):
    """Retrieve chunks and generate an extractive answer."""
    index_path = Path(index_dir)
    if not (index_path / "index.faiss").exists():
        print(f"[ERROR] FAISS index not found at {index_dir}. Run --build_index first.")
        return

    # Load index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)

    # Retrieval
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    if not results or results[0][1] > 1.5:  # Distance threshold check
        msg = "I could not find enough relevant information in the provided medical documents to answer this safely."
        lang = detect_language(query, language)
        if lang == "roman_urdu":
            msg = translate_to_roman_urdu(msg)
        print(f"\nAnswer: {msg}")
        print(DISCLAIMER)
        return

    # Simple extractive summarization (combine top chunks)
    # In a full RAG, we'd pass these to an LLM. Here we provide the best snippets.
    lang = detect_language(query, language)
    
    intro = "Based on the medical documents retrieved:"
    if lang == "roman_urdu":
        intro = translate_to_roman_urdu(intro)
    
    print(f"\nAnswer ({lang}):")
    print(intro)
    
    sources = []
    seen_content = set()
    
    for doc, score in results:
        content = doc.page_content.strip().replace("\n", " ")
        if content[:100] not in seen_content:
            print(f"- {content[:400]}...")
            seen_content.add(content[:100])
        
        source_name = Path(doc.metadata.get("source", "unknown")).name
        page_num = doc.metadata.get("page", "?")
        sources.append(f"{source_name} (Page {page_num})")

    print("\nSources used:")
    for s in sorted(list(set(sources))):
        if lang == "roman_urdu":
            print(f"  {translate_to_roman_urdu(s)}")
        else:
            print(f"  {s}")
            
    print(DISCLAIMER)

def main():
    parser = argparse.ArgumentParser(description="Medical RAG Chatbot")
    parser.add_argument("--build_index", action="store_true", help="Build FAISS index from PDFs")
    parser.add_argument("--ask", type=str, help="Question to ask the chatbot")
    parser.add_argument("--docs_dir", type=str, default="data/medical_pdfs", help="Directory containing PDFs")
    parser.add_argument("--index_dir", type=str, default="vectorstore/faiss_index", help="Directory to store FAISS index")
    parser.add_argument("--top_k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--language", type=str, choices=["english", "urdu", "roman_urdu", "auto"], default="auto", help="Answer language")
    
    args = parser.parse_args()

    if args.build_index:
        build_index(args.docs_dir, args.index_dir)
    elif args.ask:
        ask_question(args.ask, args.index_dir, args.top_k, args.language)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
