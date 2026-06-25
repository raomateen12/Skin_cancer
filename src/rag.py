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

# Medical Disclaimer
DISCLAIMER = (
    "This assistant provides educational information only. "
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
    """Basic rule-based conversion for specific phrases."""
    replacements = {
        "Source": "Zariya",
        "Page": "Safha",
        "I could not find enough relevant information": "Mujhe faraham karda medical documents mein iska sahi jawab nahi mila",
        "Based on the documents": "Documents ke mutabiq",
    }
    for eng, urd in replacements.items():
        text = text.replace(eng, urd)
    return text

def answer_question(question, language="auto", index_dir="vectorstore/faiss_index", top_k=4):
    """
    Core RAG function to be used by both CLI and Streamlit.
    Returns a dict with answer, sources, and detected language.
    """
    # Lazy imports to keep the module lightweight for simple function imports
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        return {
            "answer": f"Required libraries missing: {e}",
            "sources": [],
            "language": "english",
            "error": True
        }
    
    index_path = Path(index_dir)
    if not (index_path / "index.faiss").exists():
        return {
            "answer": f"FAISS index not found at {index_dir}. Please build the index first.",
            "sources": [],
            "language": "english",
            "error": True
        }

    # 1. Conversational Bypass (Greetings & Acknowledgements)
    clean_q = question.strip().lower().replace(".", "").replace("!", "")
    conversational_responses = {
        "ok": "Okay! Let me know if you have any questions about skin health.",
        "okay": "Okay! Let me know if you have any questions about skin health.",
        "acha": "Acha! Agar aapko skin ke baare mein koi sawal ho, toh zaroor poochein.",
        "theek hai": "Theek hai! Agar aapko skin ke baare mein koi sawal ho, toh zaroor poochein.",
        "hi": "Hello! I am the DermaLens AI Assistant. How can I help you with your skin health today?",
        "hello": "Hello! I am the DermaLens AI Assistant. How can I help you with your skin health today?",
        "hey": "Hello! I am the DermaLens AI Assistant. How can I help you with your skin health today?",
        "thanks": "You're welcome! Feel free to ask if you need more information.",
        "thank you": "You're welcome! Feel free to ask if you need more information.",
        "shukriya": "Khush aamdeed! Agar mazeed kuch poochna ho toh zaroor poochein."
    }
    
    if clean_q in conversational_responses:
        lang = detect_language(question, language)
        return {
            "answer": conversational_responses[clean_q],
            "sources": [],
            "language": lang
        }

    # Load index
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return {
            "answer": f"Error loading index: {e}",
            "sources": [],
            "language": "english",
            "error": True
        }

    # Retrieval
    results = vectorstore.similarity_search_with_score(question, k=top_k)
    lang = detect_language(question, language)
    
    if not results or results[0][1] > 1.5:  # Distance threshold check
        if lang == "roman_urdu":
            msg = (
                "Mujhe is sawal ka jawab medical documents mein nahi mila. "
                "Yeh assistant skin health aur dermatology ke baray mein sawaal answer karta hai — "
                "jaise melanoma, moles, lesion changes, ya skin cancer ke warning signs."
            )
        else:
            msg = (
                "This assistant is focused on skin health and dermatology topics. "
                "I couldn't find relevant information for that question in the connected medical documents.\n\n"
                "Try asking about:\n"
                "• Warning signs of melanoma or skin cancer\n"
                "• The ABCDE rule for moles\n"
                "• Types of skin lesions (basal cell carcinoma, actinic keratosis, etc.)\n"
                "• When to see a dermatologist\n"
                "• Sun protection and skin health"
            )
        return {
            "answer": msg,
            "sources": [],
            "language": lang
        }

    # Format answer
    intro = "Based on the medical documents retrieved:"
    if lang == "roman_urdu":
        intro = translate_to_roman_urdu(intro)
    
    answer_text = f"{intro}\n\n"
    sources = []
    seen_content = set()
    
    for doc, score in results:
        content = doc.page_content.strip().replace("\n", " ")
        if content[:100] not in seen_content:
            answer_text += f"- {content[:400]}...\n\n"
            seen_content.add(content[:100])
        
        source_name = Path(doc.metadata.get("source", "unknown")).name
        page_num = doc.metadata.get("page", "?")
        sources.append({"source": source_name, "page": page_num})

    return {
        "answer": answer_text,
        "sources": sources,
        "language": lang
    }

def ask_question(query, index_dir, top_k=4, language="auto"):
    """CLI wrapper for answer_question."""
    result = answer_question(query, language=language, index_dir=index_dir, top_k=top_k)
    
    if "error" in result:
        print(f"\n[ERROR] {result['answer']}")
        return

    print(f"\nAnswer ({result['language']}):")
    print(result['answer'])
    
    if result['sources']:
        print("\nSources used:")
        seen_sources = set()
        for s in result['sources']:
            source_str = f"{s['source']} (Page {s['page']})"
            if source_str not in seen_sources:
                if result['language'] == "roman_urdu":
                    print(f"  {translate_to_roman_urdu(source_str)}")
                else:
                    print(f"  {source_str}")
                seen_sources.add(source_str)
                
    print("\n[MEDICAL DISCLAIMER]: " + DISCLAIMER)

def build_index(docs_dir, index_dir):
    """Load PDFs, chunk text, and save FAISS index."""
    # Lazy imports
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        print(f"[ERROR] Required libraries missing: {e}")
        return

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
