import { Upload, Microscope, MessageSquare } from "lucide-react";

const steps = [
  {
    number: "01",
    icon: Upload,
    title: "Upload a clear image",
    description:
      "Take or upload a close-up photo of the skin area of concern. Ensure good lighting and a clean, unobstructed view for best results.",
    color: "#0B7FEA",
  },
  {
    number: "02",
    icon: Microscope,
    title: "Review AI-assisted insight",
    description:
      "Our model analyzes the image and returns pattern-based educational insights with confidence indicators and visual attention maps.",
    color: "#16B8C7",
  },
  {
    number: "03",
    icon: MessageSquare,
    title: "Ask follow-up questions",
    description:
      "Use the document-grounded assistant to ask follow-up questions. Answers are sourced from indexed medical literature, not generated freely.",
    color: "#7C3AED",
  },
];

export default function HowItWorksSection() {
  return (
    <section id="how-it-works" className="py-32 bg-white relative">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="text-center max-w-3xl mx-auto mb-24 space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-[#F8FAFC] border border-[#E2E8F0] rounded-full">
            <span className="text-[11px] font-semibold text-[#64748B] uppercase tracking-[0.15em]">
              How it works
            </span>
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-semibold text-[#0F172A] tracking-tight leading-tight">
            Three steps to clinical insight
          </h2>
          <p className="text-[17px] text-[#475569] font-light max-w-2xl mx-auto">
            A straightforward workflow designed to support, not replace, clinical judgment.
          </p>
        </div>

        {/* Steps */}
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {steps.map((step, i) => {
            const Icon = step.icon;
            return (
              <div
                key={step.number}
                className="relative group p-10 bg-[#F8FAFC] border border-[#0B7FEA]/40 rounded-[2rem] hover:bg-white hover:border-transparent hover:shadow-[0_8px_30px_rgba(15,23,42,0.04)] transition-all duration-300"
              >
                {/* Connector line between cards */}
                {i < steps.length - 1 && (
                  <div className="hidden md:block absolute top-[5rem] -right-4 w-8 h-[2px] bg-gradient-to-r from-[#E2E8F0] to-transparent z-10" />
                )}

                {/* Number + icon */}
                <div className="flex items-start justify-between mb-8">
                  <div
                    className="w-14 h-14 rounded-2xl flex items-center justify-center bg-white shadow-sm border border-[#E2E8F0] group-hover:scale-105 transition-transform duration-300"
                  >
                    <Icon size={24} style={{ color: step.color }} strokeWidth={1.5} />
                  </div>
                  <span
                    className="font-display text-[56px] font-bold leading-none tracking-tighter"
                    style={{ color: step.color, opacity: 0.08 }}
                  >
                    {step.number}
                  </span>
                </div>

                <h3 className="font-display text-xl font-medium text-[#0F172A] mb-3">{step.title}</h3>
                <p className="text-[#475569] text-[15px] leading-relaxed font-light">{step.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
