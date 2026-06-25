import { ScanLine, Eye, BookOpen, ShieldCheck } from "lucide-react";

const capabilities = [
  {
    icon: ScanLine,
    title: "Skin lesion pattern support",
    description:
      "Trained on HAM10000 dermoscopic images across 7 lesion classes. Provides pattern-based educational classification with probability scores.",
    color: "#0B7FEA",
  },
  {
    icon: Eye,
    title: "Explainable visual attention",
    description:
      "Grad-CAM and EigenCAM heatmaps show which regions of the image influenced the model's output, supporting interpretable review.",
    color: "#7C3AED",
  },
  {
    icon: BookOpen,
    title: "Document-grounded assistant",
    description:
      "A retrieval-based assistant answers follow-up questions from indexed medical literature. Sources are cited per chunk — no free generation.",
    color: "#16B8C7",
  },
  {
    icon: ShieldCheck,
    title: "Safety-first guidance",
    description:
      "Every result is clearly framed as educational support. Concern levels and next-step prompts prioritize safe escalation to clinical review.",
    color: "#16A34A",
  },
];

export default function CapabilitiesSection() {
  return (
    <section id="capabilities" className="py-32 bg-[#F8FAFC]">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="text-center max-w-3xl mx-auto mb-24 space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white border border-[#E2E8F0] rounded-full shadow-sm">
            <span className="text-[11px] font-semibold text-[#64748B] uppercase tracking-[0.15em]">
              Product capabilities
            </span>
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-semibold text-[#0F172A] tracking-tight leading-tight">
            Built for clinical confidence
          </h2>
          <p className="text-[17px] text-[#475569] font-light max-w-2xl mx-auto">
            Every feature is designed around transparency, interpretability, and patient safety.
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid md:grid-cols-2 gap-x-8 gap-y-12 max-w-5xl mx-auto">
          {capabilities.map((cap) => {
            const Icon = cap.icon;
            return (
              <div
                key={cap.title}
                className="group p-10 bg-white border border-[#0B7FEA]/40 rounded-[2rem] shadow-[0_4px_24px_rgba(15,23,42,0.02)] hover:shadow-[0_8px_32px_rgba(15,23,42,0.06)] hover:border-transparent transition-all duration-300 flex flex-col sm:flex-row gap-6 sm:items-start"
              >
                {/* Icon */}
                <div
                  className="w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0 bg-[#F8FAFC] border border-[#E2E8F0] group-hover:scale-105 transition-transform duration-300"
                >
                  <Icon size={24} style={{ color: cap.color }} strokeWidth={1.5} />
                </div>

                {/* Text */}
                <div className="space-y-3 pt-1">
                  <h3 className="font-display text-xl font-medium text-[#0F172A]">{cap.title}</h3>
                  <p className="text-[15px] text-[#475569] font-light leading-relaxed">{cap.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
