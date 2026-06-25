import { AlertTriangle, Camera, Stethoscope, Brain, Check, X } from "lucide-react";
import clsx from "clsx";

const abcde = [
  {
    letter: "A",
    label: "Asymmetry",
    description: "One half of the mole or birthmark does not match the other. Normal moles are typically symmetrical.",
  },
  {
    letter: "B",
    label: "Border",
    description: "Edges are ragged, notched, blurred, or irregular. The pigment may spread into surrounding skin.",
  },
  {
    letter: "C",
    label: "Color",
    description: "Varied shades of brown, black, pink, red, white, or blue within the same lesion.",
  },
  {
    letter: "D",
    label: "Diameter",
    description: "Larger than 6 mm (about the size of a pencil eraser), though melanomas can be smaller.",
  },
  {
    letter: "E",
    label: "Evolving",
    description: "Any change in size, shape, color, or new symptoms such as bleeding, itching, or crusting.",
  },
];

const aiLimitations = [
  { can: "Identify visual patterns similar to training data", canDo: true },
  { can: "Provide educational probability estimates", canDo: true },
  { can: "Highlight areas of visual attention", canDo: true },
  { can: "Replace a dermatologist examination", canDo: false },
  { can: "Diagnose skin cancer definitively", canDo: false },
  { can: "Account for patient history or symptoms", canDo: false },
];

export default function GuidePanel() {
  return (
    <div className="space-y-12 animate-fade-in">
      {/* Header */}
      <div>
        <h2 className="font-display text-2xl font-semibold text-[#0F172A] tracking-tight">Patient Guide</h2>
        <p className="text-[14px] text-[#64748B] mt-1.5 leading-relaxed">
          Understanding skin health, warning signs, and how to use DermaLens AI effectively.
        </p>
      </div>

      {/* Grid Layout for Main Content */}
      <div className="grid lg:grid-cols-[1fr_300px] gap-8">
        
        {/* Left Column: Core Educational Content */}
        <div className="space-y-12">
          
          {/* ABCDE Section */}
          <section className="space-y-4">
            <div className="flex items-center gap-2.5 mb-5">
              <div className="w-8 h-8 rounded-lg bg-[#F8FAFC] border border-[#E2E8F0] flex items-center justify-center shadow-sm">
                <AlertTriangle size={16} className="text-[#0B7FEA]" />
              </div>
              <h3 className="font-display text-[18px] font-medium text-[#0F172A]">ABCDE Warning Signs</h3>
            </div>
            
            <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] shadow-soft overflow-hidden">
              {abcde.map((item, idx) => (
                <div 
                  key={item.letter} 
                  className={clsx(
                    "flex items-start gap-5 p-5",
                    idx !== abcde.length - 1 ? "border-b border-[#F1F5F9]" : ""
                  )}
                >
                  <div className="font-display text-2xl font-bold text-[#CBD5E1] w-8 flex-shrink-0 text-center">
                    {item.letter}
                  </div>
                  <div>
                    <h4 className="font-display text-[15px] font-medium text-[#0F172A]">{item.label}</h4>
                    <p className="text-[13px] text-[#475569] mt-1.5 leading-relaxed">{item.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* System Capabilities */}
          <section className="space-y-4 pt-4">
            <div className="flex items-center gap-2.5 mb-5">
              <div className="w-8 h-8 rounded-lg bg-[#F8FAFC] border border-[#E2E8F0] flex items-center justify-center shadow-sm">
                <Brain size={16} className="text-[#10B981]" />
              </div>
              <h3 className="font-display text-[18px] font-medium text-[#0F172A]">System Capabilities</h3>
            </div>
            
            <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] shadow-soft p-6">
              <div className="grid sm:grid-cols-2 gap-x-8 gap-y-4">
                {aiLimitations.map((item, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <div className="mt-0.5">
                      {item.canDo ? (
                        <Check size={16} className="text-[#10B981]" />
                      ) : (
                        <X size={16} className="text-[#EF4444]" />
                      )}
                    </div>
                    <p className={clsx(
                      "text-[13px] leading-relaxed",
                      item.canDo ? "text-[#0F172A]" : "text-[#475569]"
                    )}>
                      {item.can}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </section>

        </div>

        {/* Right Column: Sidebar advice */}
        <div className="space-y-6">
          
          {/* When to see doctor */}
          <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] p-6 shadow-soft card-hover">
            <div className="flex items-center gap-2.5 mb-4">
              <Stethoscope size={16} className="text-[#EF4444]" />
              <h3 className="font-display text-[15px] font-medium text-[#0F172A]">Consult a doctor if:</h3>
            </div>
            <ul className="space-y-3.5">
              {[
                "A lesion is new and growing rapidly.",
                "A mole changes in size, shape, or color.",
                "A lesion bleeds, itches, or crusts.",
                "A sore does not heal within 4 weeks.",
              ].map((item, idx) => (
                <li key={idx} className="flex items-start gap-3">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#EF4444] mt-1.5 flex-shrink-0" />
                  <p className="text-[13px] text-[#475569] leading-relaxed">{item}</p>
                </li>
              ))}
            </ul>
          </div>

          {/* Imaging tips */}
          <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] p-6 shadow-soft card-hover">
            <div className="flex items-center gap-2.5 mb-4">
              <Camera size={16} className="text-[#0B7FEA]" />
              <h3 className="font-display text-[15px] font-medium text-[#0F172A]">Image guidelines</h3>
            </div>
            <ul className="space-y-3.5">
              {[
                "Use bright, diffuse natural light.",
                "Hold camera 15–20 cm from the lesion.",
                "Ensure the lesion fills 60% of the frame.",
                "Keep the skin surface dry and flat.",
              ].map((item, idx) => (
                <li key={idx} className="flex items-start gap-3">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#E2E8F0] mt-1.5 flex-shrink-0" />
                  <p className="text-[12px] text-[#64748B] leading-relaxed">{item}</p>
                </li>
              ))}
            </ul>
          </div>

        </div>
      </div>
    </div>
  );
}
