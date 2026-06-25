import { AlertTriangle } from "lucide-react";

interface SafetyNoteProps {
  compact?: boolean;
}

export default function SafetyNote({ compact = false }: SafetyNoteProps) {
  if (compact) {
    return (
      <div className="flex items-center gap-1.5 md:gap-2 px-2 md:px-3 py-1.5 md:py-2 bg-[#FFFBEB] border border-[#FEF08A] rounded-lg max-w-[200px] sm:max-w-none">
        <AlertTriangle size={12} className="text-[#F59E0B] flex-shrink-0" />
        <p className="text-[10px] md:text-[11px] text-[#92400E] font-medium leading-tight line-clamp-2 sm:line-clamp-1">
          <span className="sm:hidden">Educational support only</span>
          <span className="hidden sm:inline">Educational support only. Not a substitute for professional medical advice.</span>
        </p>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 px-4 py-3 bg-[#FFFBEB] border border-[#FEF08A] rounded-xl">
      <AlertTriangle size={16} className="text-[#F59E0B] flex-shrink-0 mt-0.5" />
      <div>
        <p className="text-sm font-semibold text-[#92400E]">Educational use only</p>
        <p className="text-xs text-[#B45309] mt-0.5 leading-relaxed">
          All insights provided by DermaLens AI are for educational and informational purposes only.
          This is not a medical diagnosis. Consult a qualified dermatologist for any skin concern.
        </p>
      </div>
    </div>
  );
}
