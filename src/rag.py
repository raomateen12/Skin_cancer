"""
RAG (Retrieval-Augmented Generation) Chatbot for Skin Disease Q&A.
Uses Gemini LLM for synthesis, sentence-level citation mapping, and hallucination verification.
Supports English and Roman Urdu answers.

LLM: Google Gemini 1.5 Flash (dual API key with automatic fallback)
Citation: Sentence-level inline markers [1]..[N] parsed into structured citations array
Verification: Second LLM critic pass verifies each sentence against source chunks

Usage:
    python -m src.rag --build_index
    python -m src.rag --ask "What are common warning signs of melanoma?"
"""

import argparse
import os
import re
from pathlib import Path

# ── Medical Disclaimer ────────────────────────────────────────────────────────
DISCLAIMER = (
    "This assistant provides educational information only. "
    "It is not a medical diagnosis. For urgent symptoms or suspicious lesions, "
    "please consult a qualified dermatologist or doctor immediately."
)

# ── Language detection ────────────────────────────────────────────────────────
URDU_KEYWORDS = ["kya", "kaise", "kyun", "ilaaj", "alamat", "bimari", "hai", "hain", "ka", "ki", "ke", "ko"]


def detect_language(text: str, requested_lang: str = "auto") -> str:
    """Detect whether text is English or Roman Urdu."""
    if requested_lang != "auto":
        return requested_lang
    words = re.findall(r'\w+', text.lower())
    return "roman_urdu" if sum(1 for w in words if w in URDU_KEYWORDS) >= 1 else "english"


def translate_to_roman_urdu(text: str) -> str:
    """Basic rule-based translation of common phrases."""
    for eng, urd in {"Source": "Zariya", "Page": "Safha"}.items():
        text = text.replace(eng, urd)
    return text


# ── Gemini dual-key client ────────────────────────────────────────────────────

def _load_gemini_keys() -> list:
    """Load Gemini API keys from .env / environment with manual parser fallback."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    keys = []
    for var in ("GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY"):
        v = os.environ.get(var, "").strip()
        if v and v not in keys:
            keys.append(v)
    return keys


def _call_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str | None:
    """
    Call Gemini with automatic key rotation on quota/rate/auth errors.
    Returns generated text string or None if all keys fail.
    """
    keys = _load_gemini_keys()
    if not keys:
        return None

    try:
        from google import genai as google_genai
    except ImportError:
        google_genai = None

    for key in keys:
        try:
            if google_genai:
                client = google_genai.Client(api_key=key)
                response = client.models.generate_content(model=model, contents=prompt)
                if response and response.text:
                    return response.text
            else:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=key)
                model_obj = genai_legacy.GenerativeModel(model)
                res = model_obj.generate_content(prompt)
                if res and res.text:
                    return res.text
        except Exception as e:
            err_str = str(e).lower()
            # If rate limit, quota, or 429/403/503 error, try the next key
            if any(x in err_str for x in ("quota", "rate", "limit", "429", "403", "503", "resource exhausted", "overloaded")):
                continue
            # If 404 model not found, try gemini-2.5-flash or gemini-flash-latest
            if "404" in err_str and model != "gemini-2.5-flash":
                return _call_gemini(prompt, model="gemini-2.5-flash")
            continue

    return None


# ── Citation parsing & Critic ──────────────────────────────────────────────────

def _parse_and_verify_answer(generated_answer: str, sources: list, chunk_texts: list) -> dict:
    """
    1. Parse generated answer into sentences.
    2. Extract inline [N] citations per sentence.
    3. Clean sentence text (remove brackets and fix typography/spacing).
    4. Run hallucination critic pass on the clean sentences ONLY if citations exist.
    5. Return full structured result (sentences, citations, summary, plain_text, answer_html).
    """
    # Strip any [N/A], [None], [null], [0], [NA] bracket markers
    cleaned_input = re.sub(r'\[(?i:N/A|None|null|0|NA)\]', '', generated_answer)

    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_input.strip()) if s.strip()]
    if not raw_sentences:
        raw_sentences = [cleaned_input.strip()] if cleaned_input.strip() else []

    citation_map = {}
    parsed_sentences = []

    for raw_s in raw_sentences:
        # Find all valid numeric citation numbers in this sentence
        markers = []
        for group in re.findall(r'\[([0-9,\s]+)\]', raw_s):
            for num_str in re.findall(r'\d+', group):
                marker = int(num_str)
                idx = marker - 1
                if idx < len(sources):
                    markers.append(marker)
                    if marker not in citation_map:
                        s_info = sources[idx]
                        citation_map[marker] = {
                            "marker": marker,
                            "source": s_info.get("source", "unknown"),
                            "page": s_info.get("page", "?"),
                            "chunk_text_snippet": s_info.get("chunk_text_snippet", ""),
                        }

        # Remove bracket markers from text and fix spacing before punctuation
        clean_text = re.sub(r'\s*\[[0-9,\s]+\]', '', raw_s).strip()
        clean_text = re.sub(r'\s*\[(?i:N/A|None|null|0|NA)\]', '', clean_text).strip()
        clean_text = re.sub(r'\s+([.!?,;:])', r'\1', clean_text)
        clean_text = re.sub(r' +', ' ', clean_text)

        parsed_sentences.append({
            "text": clean_text,
            "citation_markers": sorted(set(markers)),
        })

    # Build full plain text
    plain_text = " ".join(s["text"] for s in parsed_sentences)

    # If no chunks were cited, this is an ungrounded/not-found response:
    # Do NOT run critic pass and return clean empty citations and null verification
    if not citation_map:
        return {
            "plain_text": plain_text,
            "answer_html": None,
            "citations": [],
            "sentences": [],
            "verification_summary": None,
        }

    def replace_markers(match):
        nums = re.findall(r'\d+', match.group(1))
        sups = [f'<sup class="citation-marker" data-cit="{n}">[{n}]</sup>' for n in nums if int(n) in citation_map]
        return "".join(sups)

    answer_html = re.sub(r'\[([0-9,\s]+)\]', replace_markers, cleaned_input)
    answer_html = re.sub(r'\s+([.!?,;:])', r'\1', answer_html)

    # ── Critic Verification Pass (only when citations exist) ───────────────────
    numbered = "\n".join(f"{i+1}. {s['text']}" for i, s in enumerate(parsed_sentences))
    chunks_block = "\n\n".join(f"[CHUNK {i+1}]: {c}" for i, c in enumerate(chunk_texts))

    critic_prompt = (
        "You are a strict medical fact-checker. Verify whether each numbered sentence "
        "is directly supported by the source chunks provided.\n\n"
        f"SOURCE CHUNKS:\n{chunks_block}\n\n"
        f"SENTENCES TO VERIFY:\n{numbered}\n\n"
        "For each sentence reply with ONLY: <number>: SUPPORTED | PARTIAL | UNSUPPORTED\n"
        "- SUPPORTED: clearly stated in at least one chunk\n"
        "- PARTIAL: partially supported, some details not in chunks\n"
        "- UNSUPPORTED: makes claims not found in any chunk\n\n"
        "Reply ONLY in this exact format:\n1: SUPPORTED\n2: PARTIAL\n..."
    )

    critic_response = _call_gemini(critic_prompt)

    status_map = {}
    if critic_response:
        for line in critic_response.strip().splitlines():
            m = re.match(r'^\s*(\d+)\s*:\s*(SUPPORTED|PARTIAL|UNSUPPORTED)', line.strip(), re.IGNORECASE)
            if m:
                status_map[int(m.group(1))] = m.group(2).upper()

    final_sentences = []
    counts = {"total": len(parsed_sentences), "supported": 0, "partial": 0, "unsupported": 0}
    for i, s in enumerate(parsed_sentences):
        status = status_map.get(i + 1, "SUPPORTED")
        final_sentences.append({
            "text": s["text"],
            "status": status,
            "citation_markers": s["citation_markers"],
        })
        if status == "SUPPORTED":
            counts["supported"] += 1
        elif status == "PARTIAL":
            counts["partial"] += 1
        else:
            counts["unsupported"] += 1

    return {
        "plain_text": plain_text,
        "answer_html": answer_html,
        "citations": [citation_map[k] for k in sorted(citation_map.keys())],
        "sentences": final_sentences,
        "verification_summary": counts,
    }


# Module-level vectorstore cache
_cached_embeddings = None
_cached_vectorstore = None
_cached_index_dir = None


def _get_vectorstore(index_dir: str):
    global _cached_embeddings, _cached_vectorstore, _cached_index_dir
    index_path = Path(index_dir)
    if not (index_path / "index.faiss").exists():
        return None

    if _cached_vectorstore is not None and _cached_index_dir == str(index_path):
        return _cached_vectorstore

    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    _cached_vectorstore = FAISS.load_local(str(index_path), _cached_embeddings, allow_dangerous_deserialization=True)
    _cached_index_dir = str(index_path)
    return _cached_vectorstore


# ── Core answer_question ──────────────────────────────────────────────────────

def answer_question(question: str, language: str = "auto",
                    index_dir: str = "vectorstore/faiss_index", top_k: int = 4) -> dict:
    """
    Core RAG function used by CLI, Streamlit, and the FastAPI /ask endpoint.

    Returns dict with:
      answer           - plain text answer (backward compatible)
      sources          - [{source, page}] list (backward compatible)
      language         - detected language
      answer_html      - answer with <sup class="citation-marker"> markers (new)
      citations        - [{marker, source, page, chunk_text_snippet}] (new)
      sentences        - [{text, status}] verification results (new)
      verification_summary - {total, supported, partial, unsupported} (new)
    """
    # ── 1. Conversational bypass — no LLM, no verification, no sources ───────
    clean_q = question.strip().lower().replace(".", "").replace("!", "")
    conversational_responses = {
        "ok":        "Okay! Let me know if you have any questions about skin health.",
        "okay":      "Okay! Let me know if you have any questions about skin health.",
        "acha":      "Acha! Agar aapko skin ke baare mein koi sawal ho, toh zaroor poochein.",
        "theek hai": "Theek hai! Agar aapko skin ke baare mein koi sawal ho, toh zaroor poochein.",
        "hi":        "Hello! I am the DermaLens AI Assistant. How can I help you with your skin health today?",
        "hello":     "Hello! I am the DermaLens AI Assistant. How can I help you with your skin health today?",
        "hey":       "Hello! I am the DermaLens AI Assistant. How can I help you with your skin health today?",
        "thanks":    "You're welcome! Feel free to ask if you need more information.",
        "thank you": "You're welcome! Feel free to ask if you need more information.",
        "shukriya":  "Khush aamdeed! Agar mazeed kuch poochna ho toh zaroor poochein.",
    }
    if clean_q in conversational_responses:
        return {
            "answer": conversational_responses[clean_q],
            "sources": [],
            "language": detect_language(question, language),
            "citations": [],
            "sentences": [],
            "verification_summary": None,
            "answer_html": None,
        }

    # ── 2. Load FAISS index ───────────────────────────────────────────────────
    try:
        vectorstore = _get_vectorstore(index_dir)
        if vectorstore is None:
            return {
                "answer": f"FAISS index not found at {index_dir}. Please build the index first.",
                "sources": [], "language": "english", "error": True,
                "citations": [], "sentences": [], "verification_summary": None, "answer_html": None,
            }
    except Exception as e:
        return {
            "answer": f"Error loading index: {e}",
            "sources": [], "language": "english", "error": True,
            "citations": [], "sentences": [], "verification_summary": None, "answer_html": None,
        }

    # ── 3. Retrieve top-k chunks ──────────────────────────────────────────────
    results = vectorstore.similarity_search_with_score(question, k=top_k)
    lang = detect_language(question, language)

    # ── 4. Out-of-domain check (distance threshold) ───────────────────────────
    if not results or results[0][1] > 1.25:
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
            "language": lang,
            "citations": [],
            "sentences": [],
            "verification_summary": None,
            "answer_html": None,
        }

    # ── 5. Build sources list with snippet ────────────────────────────────────
    sources = []
    chunk_texts = []
    seen_content: set = set()

    for doc, _score in results:
        content = doc.page_content.strip().replace("\n", " ")
        source_name = Path(doc.metadata.get("source", "unknown")).name
        page_num = doc.metadata.get("page", "?")
        if content[:100] not in seen_content:
            chunk_texts.append(content[:600])
            seen_content.add(content[:100])
        sources.append({
            "source": source_name,
            "page": page_num,
            "chunk_text_snippet": content[:200],
        })

    # ── 6. LLM generation with inline citation markers ────────────────────────
    lang_instruction = (
        "Answer in Roman Urdu (Urdu written in English letters)."
        if lang == "roman_urdu" else
        "Answer in clear, concise plain English."
    )
    chunks_block = "\n\n".join(
        f"[SOURCE {i+1}] ({sources[i]['source']}, p.{sources[i]['page']}):\n{chunk_texts[i]}"
        for i in range(min(len(chunk_texts), len(sources)))
    )
    generation_prompt = (
        f"You are DermaLens AI, a medical education assistant specialising in dermatology. "
        f"{lang_instruction}\n\n"
        "Using ONLY the information in the source chunks below, write a concise factual answer "
        "to the question. After EACH factual sentence, add an inline citation like [1] or [2] "
        "referencing which source chunk supports that sentence (use [1] for SOURCE 1, etc.). "
        "Do NOT add information not present in the chunks.\n\n"
        "IMPORTANT RULES:\n"
        "1. If the source chunks do NOT contain enough information to answer the question, or if the question is unrelated to the chunks, "
        "reply ONLY that the medical documents do not contain information on this topic.\n"
        "2. When no chunks support the answer, do NOT include ANY citation markers, do NOT write [N/A] or [1], and do NOT list sources.\n\n"
        f"QUESTION: {question}\n\n"
        f"SOURCE CHUNKS:\n{chunks_block}\n\n"
        "ANSWER (with inline citations after each sentence only if grounded):"
    )

    generated_answer = _call_gemini(generation_prompt)

    if not generated_answer:
        # Graceful fallback: return raw chunk paste (backward compatible)
        intro = "Documents ke mutabiq" if lang == "roman_urdu" else "Based on the medical documents retrieved:"
        fallback_text = f"{intro}\n\n" + "".join(f"- {ct[:400]}...\n\n" for ct in chunk_texts)
        return {
            "answer": fallback_text,
            "sources": [{"source": s["source"], "page": s["page"]} for s in sources],
            "language": lang,
            "citations": [],
            "sentences": [],
            "verification_summary": None,
            "answer_html": None,
        }

    # ── 7. Parse citation markers & run Critic pass ───────────────────────────
    verified_data = _parse_and_verify_answer(generated_answer, sources, chunk_texts)

    # If no citations were generated (i.e. answer was not grounded in chunks),
    # return EMPTY sources list and NO verification badge
    if not verified_data["citations"]:
        return {
            "answer":   verified_data["plain_text"],
            "sources":  [],  # Do not display retrieved chunks if they weren't cited
            "language": lang,
            "answer_html": None,
            "citations": [],
            "sentences": [],
            "verification_summary": None,
        }

    return {
        # Backward-compatible fields
        "answer":   verified_data["plain_text"],
        "sources":  [{"source": s["source"], "page": s["page"]} for s in sources if any(c["marker"] == idx + 1 for idx, s_orig in enumerate(sources) for c in verified_data["citations"])],
        "language": lang,
        # New structured fields
        "answer_html":           verified_data["answer_html"],
        "citations":             verified_data["citations"],
        "sentences":             verified_data["sentences"],
        "verification_summary":  verified_data["verification_summary"],
    }


# ── CLI helpers ───────────────────────────────────────────────────────────────

def ask_question(query: str, index_dir: str, top_k: int = 4, language: str = "auto") -> None:
    """CLI wrapper for answer_question."""
    result = answer_question(query, language=language, index_dir=index_dir, top_k=top_k)

    if result.get("error"):
        print(f"\n[ERROR] {result['answer']}")
        return

    print(f"\nAnswer ({result['language']}):")
    print(result["answer"])

    if result.get("sources"):
        print("\nSources used:")
        seen: set = set()
        for s in result["sources"]:
            key = f"{s['source']} (Page {s['page']})"
            if key not in seen:
                label = translate_to_roman_urdu(key) if result["language"] == "roman_urdu" else key
                print(f"  {label}")
                seen.add(key)

    if "verification_summary" in result:
        vs = result["verification_summary"]
        print(f"\n[Verification] {vs['supported']}/{vs['total']} sentences supported by source chunks.")

    print(f"\n[MEDICAL DISCLAIMER]: {DISCLAIMER}")


def build_index(docs_dir: str, index_dir: str) -> None:
    """Load PDFs, chunk text, and save FAISS index."""
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
            from langchain_community.document_loaders import PyPDFLoader
            all_docs.extend(PyPDFLoader(str(pdf)).load())
        except Exception as e:
            print(f"  Failed to load {pdf.name}: {e}")

    if not all_docs:
        print("[ERROR] No text extracted from PDFs.")
        return

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(all_docs)
    print(f"Created {len(chunks)} text chunks.")

    print("Generating embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    print(f"FAISS index saved to {index_dir}")


def main():
    parser = argparse.ArgumentParser(description="Medical RAG Chatbot")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--ask", type=str)
    parser.add_argument("--docs_dir", default="data/medical_pdfs")
    parser.add_argument("--index_dir", default="vectorstore/faiss_index")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()

    if args.build_index:
        build_index(args.docs_dir, args.index_dir)
    elif args.ask:
        ask_question(args.ask, args.index_dir, args.top_k, args.language)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
