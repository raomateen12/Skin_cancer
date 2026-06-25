import { ShieldCheck, GraduationCap, Scale, Stethoscope, AlertCircle } from "lucide-react";

const trustPoints = [
  {
    icon: GraduationCap,
    title: "Educational support only",
    description:
      "A research-backed tool providing pattern-based insights to support learning and awareness, not clinical decision-making.",
  },
  {
    icon: ShieldCheck,
    title: "Not a medical diagnosis",
    description:
      "No output constitutes a diagnosis, treatment recommendation, or medical opinion. Results are strictly educational.",
  },
  {
    icon: Stethoscope,
    title: "Consult a dermatologist",
    description:
      "For any suspicious, changing, or concerning skin lesion, always consult a qualified healthcare provider promptly.",
  },
  {
    icon: Scale,
    title: "Fairness-aware research layer",
    description:
      "Built with group-level fairness monitoring across demographic attributes, utilizing transparent explainability models.",
  },
];

const urgentSigns = [
  "Rapidly changing mole or lesion",
  "Bleeding, itching, or crusting",
  "Asymmetric or irregular border",
  "Multiple colors in a single lesion",
  "Lesion larger than 6 mm",
];

export default function SafetySection() {
  return (
    <section id="safety" className="py-32 bg-white relative overflow-hidden">
      {/* Very subtle top border gradient */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#E2E8F0] to-transparent" />
      
      <div className="max-w-7xl mx-auto px-6">
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center mb-24 space-y-6 animate-fade-up">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-[#F8FAFC] border border-[#E2E8F0] rounded-full">
            <span className="text-[11px] font-semibold text-[#64748B] tracking-[0.15em] uppercase">
              Safety & Trust
            </span>
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-semibold text-[#0F172A] leading-tight tracking-tight">
            Designed around patient safety <br className="hidden md:block" />
            <span className="text-[#64748B]">and clinical trust.</span>
          </h2>
        </div>

        {/* 2x2 Grid of Core Principles */}
        <div className="grid md:grid-cols-2 gap-x-12 gap-y-16 mb-24 max-w-5xl mx-auto">
          {trustPoints.map((point) => {
            const Icon = point.icon;
            return (
              <div key={point.title} className="flex items-start gap-6 group">
                <div className="w-14 h-14 rounded-2xl bg-[#F8FAFC] border border-[#E2E8F0] flex items-center justify-center flex-shrink-0 group-hover:bg-white group-hover:shadow-sm group-hover:border-[#CBD5E1] transition-all duration-300">
                  <Icon size={24} className="text-[#0B7FEA]" strokeWidth={1.5} />
                </div>
                <div className="space-y-2.5 pt-1">
                  <h3 className="font-display text-[20px] font-medium text-[#0F172A]">
                    {point.title}
                  </h3>
                  <p className="text-[15px] text-[#475569] leading-relaxed font-light">
                    {point.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Bottom Clinical Notice Box */}
        <div className="max-w-5xl mx-auto bg-[#F8FAFC] rounded-[2rem] p-10 md:p-14 border border-[#0B7FEA]/40 shadow-[0_4px_24px_rgba(15,23,42,0.02)] hover:border-transparent hover:shadow-[0_8px_32px_rgba(15,23,42,0.06)] transition-all duration-300">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-20">
            {/* Left side: Important Notice */}
            <div className="space-y-6">
              <div className="flex items-center gap-3 mb-2">
                <AlertCircle size={22} className="text-[#F59E0B]" strokeWidth={2} />
                <h3 className="font-display text-[22px] font-medium text-[#0F172A]">
                  Important Notice
                </h3>
              </div>
              <p className="text-[15px] text-[#475569] leading-relaxed">
                DermaLens AI provides educational pattern-recognition insights based on dermoscopic image analysis. It <strong className="text-[#0F172A] font-medium">does not provide medical diagnoses</strong> and is not a substitute for professional dermatological evaluation.
              </p>
              <p className="text-[15px] text-[#475569] leading-relaxed">
                Always consult a qualified clinician for any skin concern or if you are unsure about a lesion.
              </p>
            </div>

            {/* Right side: Seek Care */}
            <div className="lg:border-l border-[#E2E8F0] lg:pl-16 space-y-6">
              <div className="flex items-center gap-3 mb-2">
                <Stethoscope size={22} className="text-[#16A34A]" strokeWidth={2} />
                <h3 className="font-display text-[22px] font-medium text-[#0F172A]">
                  When to seek immediate care
                </h3>
              </div>
              <ul className="space-y-3.5">
                {urgentSigns.map((sign) => (
                  <li key={sign} className="flex items-start gap-3.5 group">
                    <span className="mt-2.5 w-1.5 h-1.5 rounded-full bg-[#16A34A]/30 group-hover:bg-[#16A34A] transition-colors duration-300 flex-shrink-0" />
                    <span className="text-[15px] text-[#475569] leading-relaxed">{sign}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

      </div>
    </section>
  );
}
