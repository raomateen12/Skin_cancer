"use client";

import { useState, useRef, useEffect } from "react";
import { Send, WifiOff, Globe, FileText, ArrowRight } from "lucide-react";
import clsx from "clsx";
import { askAssistant } from "@/lib/api";
import SafetyNote from "@/components/shared/SafetyNote";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Array<{ source: string; page: number }>;
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
            sources: result.sources,
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
            Document-grounded Q&amp;A referencing indexed dermatology literature.
          </p>
        </div>
      </div>

      <div className="mb-4">
        <SafetyNote compact />
      </div>

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto space-y-6 pb-[180px] md:pb-32 scrollbar-hide">
        {/* Empty state / Offline state */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-start pt-4 sm:pt-0 sm:justify-center h-full min-h-[400px]">
            {unavailable ? (
              /* Offline card */
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
              /* Normal empty state */
              <>
                <div className="w-16 h-16 rounded-2xl bg-white border border-[#E2E8F0] shadow-soft flex items-center justify-center mb-6">
                  <FileText size={24} className="text-[#0B7FEA]" />
                </div>
                <h3 className="font-display text-[18px] font-medium text-[#0F172A] mb-2 tracking-tight">How can I help?</h3>
                <p className="text-[13px] text-[#64748B] text-center max-w-sm mb-8 leading-relaxed">
                  Ask questions about skin conditions, warning signs, or next steps. All answers are grounded in provided medical documents.
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
                <span className="break-words" style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{msg.content}</span>
              </div>

              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
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
