import { UploadCloud, CheckCircle2, Search, Info } from "lucide-react";

export default function ProductMockup() {
  return (
    <div className="relative w-full max-w-[500px] mx-auto lg:ml-auto">
      {/* Outer soft glow instead of aggressive gradient */}
      <div className="absolute -inset-8 bg-[#0B7FEA]/5 rounded-[3rem] blur-2xl pointer-events-none" />

      {/* Main Browser/App Frame */}
      <div className="relative bg-[#F8FBFF] rounded-[24px] border border-[#E5EAF0] shadow-[0_12px_40px_-12px_rgba(15,23,42,0.08)] overflow-hidden">
        
        {/* Top Browser Bar */}
        <div className="flex items-center justify-between px-5 py-3.5 bg-white border-b border-[#E5EAF0]">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-[#E2E8F0]" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#E2E8F0]" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#E2E8F0]" />
          </div>
          <div className="flex-1 flex justify-center">
            <div className="px-4 py-1.5 bg-[#F5F7FB] rounded-md border border-[#E5EAF0] text-[11px] text-[#64748B] font-mono tracking-wide">
              dermalens.ai/analyze
            </div>
          </div>
          <div className="flex items-center">
            <span className="px-2.5 py-1 bg-[#F0FDF4] text-[#16A34A] border border-[#BBF7D0] text-[10px] font-semibold rounded-full shadow-sm">
              Clinical support
            </span>
          </div>
        </div>

        <div className="p-5 space-y-4">
          
          {/* Main Content Area (Split Left/Right) */}
          <div className="flex flex-col sm:flex-row gap-4">
            
            {/* Left Card: Image Review */}
            <div className="flex-1 bg-white rounded-2xl p-4 border border-[#E5EAF0] shadow-sm flex flex-col">
              <h3 className="font-display text-[13px] font-semibold text-[#0F172A] mb-3">
                Image review
              </h3>
              <div className="flex-1 border border-dashed border-[#CBD5E1] rounded-xl bg-[#F8FBFF] p-4 flex flex-col items-center justify-center text-center gap-2">
                <div className="w-10 h-10 rounded-full bg-[#EFF6FF] flex items-center justify-center mb-1">
                  <UploadCloud size={18} className="text-[#0B7FEA]" />
                </div>
                <p className="text-[12px] font-medium text-[#0F172A]">
                  Upload a clear skin image
                </p>
                <p className="text-[10px] text-[#64748B] leading-tight max-w-[140px]">
                  Good lighting &bull; Centered lesion &bull; JPG/PNG
                </p>
              </div>
            </div>

            {/* Right Card: Review Guidance */}
            <div className="flex-1 bg-white rounded-2xl p-4 border border-[#E5EAF0] shadow-sm">
              <h3 className="font-display text-[13px] font-semibold text-[#0F172A] mb-3">
                Review guidance
              </h3>
              <ul className="space-y-3.5">
                {[
                  "Check recent changes",
                  "Compare size and color",
                  "Consult a dermatologist if concerned",
                ].map((item, idx) => (
                  <li key={idx} className="flex items-start gap-2.5">
                    <CheckCircle2 size={14} className="text-[#16B8C7] flex-shrink-0 mt-0.5" strokeWidth={2.5} />
                    <span className="text-[11px] text-[#475569] leading-snug">
                      {item}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
            
          </div>

          {/* Lower Card: Assistant Note */}
          <div className="bg-white rounded-2xl p-4 border border-[#E5EAF0] shadow-sm">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="font-display text-[13px] font-semibold text-[#0F172A]">
                Assistant note
              </h3>
            </div>
            <p className="text-[12px] text-[#64748B] mb-3">
              Ask follow-up questions using document-grounded dermatology guidance.
            </p>
            {/* Input Bar Preview */}
            <div className="flex items-center gap-2.5 bg-[#F8FBFF] border border-[#E5EAF0] rounded-lg px-3 py-2">
              <Search size={14} className="text-[#94A3B8]" />
              <span className="text-[11px] text-[#94A3B8]">
                Ask about warning signs...
              </span>
            </div>
          </div>

          {/* Small Safety Strip */}
          <div className="bg-[#FFFBEB] border border-[#FEF08A] rounded-xl px-3 py-2 flex items-center gap-2">
            <Info size={14} className="text-[#D97706]" />
            <span className="text-[10px] font-medium text-[#92400E]">
              Educational support only &mdash; not a diagnosis.
            </span>
          </div>

        </div>
      </div>
    </div>
  );
}
