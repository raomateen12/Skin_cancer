"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send, WifiOff, Globe, FileText, ArrowRight,
  CheckCircle, AlertTriangle, ShieldAlert, Info
} from "lucide-react";
import clsx from "clsx";
import { askAssistant } from "@/lib/api";
import SafetyNote from "@/components/shared/SafetyNote";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Citation {
  marker: number;
  source: string;
  page: number | string;
  chunk_text_snippet: string;
}

interface SentenceVerification {
  text: string;
  status: "SUPPORTED" | "PARTIAL" | "UNSUPPORTED";
  citation_markers?: number[];
}

interface VerificationSummary {
  total: number;
  supported: number;
  partial: number;
  unsupported: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  answer_html?: string;
  sources?: Array<{ source: string; page: number | string }>;
  citations?: Citation[];
  sentences?: SentenceVerification[];
  verification_summary?: VerificationSummary;
  language?: string;
}

const LANGUAGE_OPTIONS = [
  { value: "auto", label: "Auto-detect" },
  { value: "english", label: "English" },
  { value: "roman_urdu", label: "Roman Urdu" },
];

const SUGGESTED_QUESTIONS = [
  "What are the warning signs of melanoma?",
  "How is basal cell carcinoma treated?",
  "What is the ABCDE rule for moles?",
  "When should I see a dermatologist?",
];

// ── Citation Popup Component ──────────────────────────────────────────────────

function CitationPopup({ citation, onClose }: { citation: Citation; onClose: () => void }) {
  return (
    <div
      className="absolute z-50 bottom-full left-0 mb-1 w-72 bg-white border border-[#CBD5E1] rounded-xl shadow-[0_8px_32px_rgba(0,0,0,0.12)] p-4 text-left"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-1.5">
          <FileText size={12} className="text-[#0B7FEA] flex-shrink-0" />
          <span className="text-[11px] font-semibold text-[#0F172A]">{citation.source}</span>
        </div>
        <span className="text-[10px] text-[#64748B] font-medium whitespace-nowrap">p. {citation.page}</span>
      </div>
      {citation.chunk_text_snippet && (
        <p className="text-[11px] text-[#475569] leading-relaxed line-clamp-4 border-t border-[#F1F5F9] pt-2 mt-2">
          {citation.chunk_text_snippet}
        </p>
      )}
      <button
        onClick={onClose}
        className="absolute top-2 right-2 w-5 h-5 flex items-center justify-center rounded text-[#94A3B8] hover:text-[#475569] text-[14px] font-medium leading-none"
      >
        ×
      </button>
    </div>
  );
}

// ── Inline Citation Superscript ───────────────────────────────────────────────

function CitationMarker({ marker, citations }: { marker: number; citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  const citation = citations.find((c) => c.marker === marker);
  if (!citation) return <sup className="text-[10px] text-[#0B7FEA]">[{marker}]</sup>;

  return (
    <span className="relative inline-block">
      <sup
        className="cursor-pointer text-[#0B7FEA] hover:text-[#0ea5e9] font-semibold text-[10px] transition-colors ml-0.5"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        title={`Source: ${citation.source}, p.${citation.page}`}
      >
        [{marker}]
      </sup>
      {open && <CitationPopup citation={citation} onClose={() => setOpen(false)} />}
    </span>
  );
}

// ── Verified Answer Renderer ──────────────────────────────────────────────────

function VerifiedAnswer({
  sentences,
  citations,
  plainText,
}: {
  sentences?: SentenceVerification[];
  citations?: Citation[];
  plainText: string;
}) {
  if (!sentences || sentences.length === 0) {
    return (
      <span className="break-words" style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
        {plainText}
      </span>
    );
  }

  return (
    <span>
      {sentences.map((sent, i) => {
        const isUnsupported = sent.status === "UNSUPPORTED";
        const isPartial = sent.status === "PARTIAL";

        return (
          <span key={i} className="relative group/sent">
            <span
              className={clsx(
                "break-words",
                isUnsupported && "underline decoration-amber-400 decoration-dotted underline-offset-2 cursor-help bg-amber-50/50",
                isPartial && "underline decoration-slate-300 decoration-dotted underline-offset-2"
              )}
              title={
                isUnsupported
                  ? "Warning: This statement could not be directly verified against the source documents."
                  : isPartial
                  ? "Note: This statement is partially supported by the source documents."
                  : undefined
              }
            >
              {sent.text}
            </span>
            {/* Inline citation markers after each sentence */}
            {citations && sent.citation_markers && sent.citation_markers.length > 0 && (
              <span className="inline-flex items-center gap-0.5 ml-0.5">
                {sent.citation_markers.map((m) => (
                  <CitationMarker key={m} marker={m} citations={citations} />
                ))}
              </span>
            )}
            {isUnsupported && (
              <span className="inline-block ml-1 align-middle opacity-80" title="Unverified statement">
                <AlertTriangle size={11} className="text-amber-500 inline" />
              </span>
            )}
            {" "}
          </span>
        );
      })}
    </span>
  );
}

// ── Verification Badge ────────────────────────────────────────────────────────

function VerificationBadge({ summary }: { summary: VerificationSummary }) {
  const allSupported = summary.unsupported === 0 && summary.partial === 0;
  const hasUnsupported = summary.unsupported > 0;

  return (
    <div
      className={clsx(
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold border",
        allSupported
          ? "bg-[#F0FDF4] border-[#BBF7D0] text-[#16A34A]"
          : hasUnsupported
          ? "bg-[#FFFBEB] border-[#FDE68A] text-[#B45309]"
          : "bg-[#F8FAFC] border-[#E2E8F0] text-[#475569]"
      )}
      title={`${summary.supported} supported, ${summary.partial} partial, ${summary.unsupported} unsupported out of ${summary.total} total statements`}
    >
      {allSupported ? (
        <CheckCircle size={11} />
      ) : hasUnsupported ? (
        <ShieldAlert size={11} />
      ) : (
        <Info size={11} />
      )}
      {summary.supported}/{summary.total} statements verified
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function AssistantChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (question?: string) => {
    const q = question ?? input.trim();
    if (!q || loading) return;
    setInput("");

    const userMsg: Message = { role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const result = await askAssistant(q, language);
      if (!result || result.ok === false) {
        setUnavailable(true);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: result?.answer ?? "Knowledge base is not connected in this environment.",
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: result.answer,
            answer_html: result.answer_html,
            sources: result.sources,
            citations: result.citations ?? [],
            sentences: result.sentences ?? [],
            verification_summary: result.verification_summary,
            language: result.language_detected,
          },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "An unexpected error occurred. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full relative w-full max-w-3xl mx-auto px-4 sm:px-6 md:px-0 pt-4 sm:pt-6 md:pt-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6 flex-shrink-0">
        <div>
          <h2 className="font-display text-2xl font-semibold text-[#0F172A]">Clinical Assistant</h2>
          <p className="text-[14px] text-[#64748B] mt-1.5">
            Document-grounded Q&amp;A with citation verification.
          </p>
        </div>
      </div>

      <div className="mb-4">
        <SafetyNote compact />
      </div>

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto space-y-6 pb-[180px] md:pb-32 scrollbar-hide">
        {/* Empty / Offline state */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-start pt-4 sm:pt-0 sm:justify-center h-full min-h-[400px]">
            {unavailable ? (
              <div className="w-full max-w-md bg-white border border-[#E2E8F0] rounded-[1.25rem] p-8 shadow-soft text-center">
                <div className="w-12 h-12 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] flex items-center justify-center mx-auto mb-5 shadow-sm">
                  <WifiOff size={20} className="text-[#94A3B8]" />
                </div>
                <p className="font-display text-[16px] font-medium text-[#0F172A] mb-2">
                  Knowledge base not connected
                </p>
                <p className="text-[13px] text-[#64748B] leading-relaxed">
                  The medical knowledge assistant is not fully connected in this environment.
                </p>
                <p className="text-[12px] text-[#94A3B8] mt-4 pt-4 border-t border-[#E2E8F0] leading-relaxed">
                  Document-grounded answers will appear once the backend knowledge base is available.
                </p>
              </div>
            ) : (
              <>
                <div className="w-16 h-16 rounded-2xl bg-white border border-[#E2E8F0] shadow-soft flex items-center justify-center mb-6">
                  <FileText size={24} className="text-[#0B7FEA]" />
                </div>
                <h3 className="font-display text-[18px] font-medium text-[#0F172A] mb-2 tracking-tight">How can I help?</h3>
                <p className="text-[13px] text-[#64748B] text-center max-w-sm mb-8 leading-relaxed">
                  Ask questions about skin conditions, warning signs, or next steps. Answers include source citations and verification status.
                </p>
                <div className="w-full max-w-2xl grid sm:grid-cols-2 gap-3 sm:gap-4 px-1 sm:px-0">
                  {SUGGESTED_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      onClick={() => sendMessage(q)}
                      className="flex items-center justify-between text-left px-4 py-3.5 sm:px-5 sm:py-4 bg-white border border-[#E2E8F0] rounded-[1rem] hover:border-[#CBD5E1] shadow-sm hover:shadow-soft card-hover group"
                    >
                      <span className="text-[13px] font-medium text-[#475569] group-hover:text-[#0F172A] leading-snug pr-3">{q}</span>
                      <ArrowRight size={14} className="text-[#CBD5E1] group-hover:text-[#0B7FEA] transition-colors flex-shrink-0" />
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, i) => (
          <div key={i} className={clsx("flex gap-4 max-w-4xl", msg.role === "user" ? "ml-auto flex-row-reverse" : "")}>
            {/* Avatar */}
            <div
              className={clsx(
                "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-[10px] font-bold mt-1 shadow-sm",
                msg.role === "user"
                  ? "bg-[#0F172A] text-white"
                  : "bg-white border border-[#E2E8F0] text-[#0B7FEA]"
              )}
            >
              {msg.role === "user" ? "U" : "AI"}
            </div>

            {/* Bubble */}
            <div className={clsx("max-w-[85%] space-y-2", msg.role === "user" ? "items-end" : "items-start")}>
              <div
                className={clsx(
                  "px-5 py-4 rounded-[1.25rem] text-[14px] leading-relaxed",
                  msg.role === "user"
                    ? "bg-[#F1F5F9] text-[#0F172A] rounded-tr-[4px]"
                    : "bg-white border border-[#E2E8F0] text-[#475569] rounded-tl-[4px] shadow-soft"
                )}
              >
                {msg.role === "assistant" && msg.sentences && msg.sentences.length > 0 ? (
                  <VerifiedAnswer
                    sentences={msg.sentences}
                    citations={msg.citations}
                    plainText={msg.content}
                  />
                ) : (
                  <span className="break-words" style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                    {msg.content}
                  </span>
                )}
              </div>

              {/* Verification badge + citation markers row (ONLY if grounded with citations) */}
              {msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
                <div className="flex flex-wrap items-center gap-2 pt-0.5 pl-1">
                  {/* Verification badge */}
                  {msg.verification_summary && (
                    <VerificationBadge summary={msg.verification_summary} />
                  )}

                  {/* Citation markers as clickable chips */}
                  <div className="flex flex-wrap gap-1.5">
                    {msg.citations.map((cit) => (
                      <span key={cit.marker} className="relative group/cit">
                        <CitationMarker marker={cit.marker} citations={msg.citations!} />
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Sources (ONLY if grounded with citations) */}
              {msg.citations && msg.citations.length > 0 && msg.sources && msg.sources.length > 0 && (
                <div className="flex flex-wrap gap-2 pt-1 pl-1">
                  {msg.sources.map((src, si) => (
                    <span
                      key={si}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-white border border-[#E2E8F0] rounded-md text-[10px] text-[#64748B] font-medium shadow-sm"
                    >
                      <FileText size={10} className="text-[#94A3B8]" />
                      {src.source} (p.{src.page})
                    </span>
                  ))}
                </div>
              )}

              {/* Unsupported warning legend (only if any unsupported exist and grounded) */}
              {msg.citations && msg.citations.length > 0 && msg.verification_summary && msg.verification_summary.unsupported > 0 && (
                <div className="flex items-start gap-1.5 px-3 py-2 bg-[#FFFBEB] border border-[#FDE68A] rounded-lg text-[10px] text-[#92400E] mt-1">
                  <AlertTriangle size={10} className="text-amber-500 mt-0.5 flex-shrink-0" />
                  <span>
                    {msg.verification_summary.unsupported} statement{msg.verification_summary.unsupported > 1 ? "s" : ""} could not be
                    directly verified against source documents. Dotted underline indicates unverified content.
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Loading bubble */}
        {loading && (
          <div className="flex gap-4 max-w-4xl">
            <div className="w-8 h-8 rounded-full bg-white border border-[#E2E8F0] text-[#0B7FEA] shadow-sm flex items-center justify-center flex-shrink-0 text-[10px] font-bold mt-1">
              AI
            </div>
            <div className="px-5 py-4.5 bg-white border border-[#E2E8F0] rounded-[1.25rem] rounded-tl-[4px] shadow-soft flex items-center gap-1.5 min-h-[52px]">
              {[0, 1, 2].map((d) => (
                <div
                  key={d}
                  className="w-1.5 h-1.5 rounded-full bg-[#CBD5E1] animate-bounce"
                  style={{ animationDelay: `${d * 150}ms` }}
                />
              ))}
            </div>
          </div>
        )}

        <div ref={endRef} />
      </div>

      {/* Input Area (Fixed at bottom) */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[#F8FAFC] via-[#F8FAFC] to-transparent pt-8 pb-[84px] md:pb-2 z-20 px-4 sm:px-6 md:px-0">
        <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] shadow-soft-lg p-2 transition-shadow focus-within:shadow-[0_10px_40px_-4px_rgba(11,127,234,0.08)]">
          {/* Top row: Language settings */}
          <div className="flex items-center gap-2 px-3 pt-1 pb-2 border-b border-[#F1F5F9]">
            <Globe size={13} className="text-[#94A3B8]" />
            <span className="text-[10px] font-semibold text-[#94A3B8] uppercase tracking-wider hidden sm:inline">Language:</span>
            <div className="flex flex-wrap items-center gap-1 ml-1">
              {LANGUAGE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setLanguage(opt.value)}
                  className={clsx(
                    "px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors",
                    language === opt.value
                      ? "bg-[#F1F5F9] text-[#0F172A]"
                      : "text-[#64748B] hover:bg-[#F8FAFC] hover:text-[#0F172A]"
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Bottom row: Input */}
          <div className="flex items-end gap-2 pt-2 px-2">
            <textarea
              id="assistant-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Ask a medical question..."
              className="flex-1 max-h-32 min-h-[44px] px-2 py-3 bg-transparent text-[14px] text-[#0F172A] placeholder-[#94A3B8] focus:outline-none resize-none"
              disabled={loading}
              rows={1}
            />
            <button
              id="assistant-send"
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
              className={clsx(
                "w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 mb-0.5",
                input.trim() && !loading
                  ? "bg-[#0B7FEA] hover:bg-[#0ea5e9] text-white shadow-md hover:-translate-y-[1px]"
                  : "bg-[#F8FAFC] text-[#CBD5E1] cursor-not-allowed"
              )}
              aria-label="Send message"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
