"use client";

import { Activity, WifiOff, Clock, ChevronRight, AlertTriangle, Info, CheckCircle2 } from "lucide-react";
import StatusBadge from "@/components/shared/StatusBadge";
import clsx from "clsx";

export interface PredictResult {
  ok?: boolean;
  available: boolean;
  // Backend v2 fields
  predicted_code?: string;
  predicted_name?: string;
  concern_message?: string;
  top_predictions?: Array<{ code: string; name: string; confidence: number }>;
  // Backward compat fields
  predicted_class?: string;
  predicted_label?: string;
  confidence?: number;
  top_3?: Array<{ label: string; probability: number }>;
  concern_level?: "low" | "moderate" | "high";
  next_steps?: string[];
  // Explainability
  gradcam_available?: boolean;
  gradcam_images?: string[];
  // Error fields
  error?: string;
  missing_path?: string;
  disclaimer?: string;
}

interface ResultPanelProps {
  result: PredictResult | null;
  analyzing: boolean;
}

const CLASS_LABELS: Record<string, string> = {
  akiec: "Actinic Keratosis",
  bcc: "Basal Cell Carcinoma",
  bkl: "Benign Keratosis",
  df: "Dermatofibroma",
  mel: "Melanoma",
  nv: "Melanocytic Nevus",
  vasc: "Vascular Lesion",
};

export default function ResultPanel({ result, analyzing }: ResultPanelProps) {
  // Analyzing state
  if (analyzing) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-5 py-16">
        <div className="w-14 h-14 rounded-full bg-[#F8FAFC] border border-[#E2E8F0] shadow-sm flex items-center justify-center">
          <div className="w-6 h-6 border-[2px] border-[#CBD5E1] border-t-[#0F172A] rounded-full animate-spin" />
        </div>
        <div className="text-center">
          <p className="font-display text-[15px] font-medium text-[#0F172A]">Analyzing image...</p>
          <p className="text-[13px] text-[#64748B] mt-1.5">Processing clinical patterns</p>
        </div>
      </div>
    );
  }

  // No result yet
  if (!result) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-5 py-16">
        <div className="w-16 h-16 rounded-2xl bg-[#F8FAFC] border border-[#E2E8F0] shadow-sm flex items-center justify-center">
          <Clock size={24} className="text-[#94A3B8]" />
        </div>
        <div className="text-center space-y-2">
          <p className="font-display text-[16px] font-medium text-[#0F172A]">No analysis yet</p>
          <p className="text-[13px] text-[#64748B] max-w-[220px] mx-auto leading-relaxed">
            Upload an image and run the analysis to view clinical insights here.
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg">
          <div className="w-1.5 h-1.5 rounded-full bg-[#94A3B8]" />
          <span className="text-[11px] text-[#64748B] font-medium">System ready</span>
        </div>
      </div>
    );
  }

  // Backend not available or checkpoint missing
  if (!result.available || result.ok === false) {
    const isBackendDown = result.error?.includes("Backend is not running") || result.error?.includes("not running");
    const isCheckpointMissing = result.missing_path || result.error?.includes("weights") || result.error?.includes("checkpoint");

    return (
      <div className="space-y-4">
        <div className="p-8 bg-[#F8FAFC] border border-[#E2E8F0] rounded-[1.25rem] flex flex-col items-center gap-4 text-center">
          <div className="w-12 h-12 rounded-xl bg-white border border-[#E2E8F0] shadow-sm flex items-center justify-center">
            <WifiOff size={20} className="text-[#94A3B8]" />
          </div>
          <div>
            <p className="font-display text-[15px] font-medium text-[#0F172A]">
              {isBackendDown ? "Backend is not running" : isCheckpointMissing ? "Model checkpoint unavailable" : "Analysis unavailable"}
            </p>
            <p className="text-[13px] text-[#64748B] mt-2 leading-relaxed max-w-[300px] mx-auto">
              {isBackendDown
                ? "Start the FastAPI backend on port 8000 to enable predictions."
                : isCheckpointMissing
                ? "The model weights file is not present. Place the trained checkpoint in the checkpoints/ directory."
                : "The analysis service is not available. Ensure the backend is running and the model is loaded."
              }
            </p>
            {result.missing_path && (
              <p className="text-[11px] text-[#94A3B8] mt-2 font-mono">
                Expected: {result.missing_path}
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Full result
  const label = result.predicted_label ?? (result.predicted_class ? CLASS_LABELS[result.predicted_class] ?? result.predicted_class : "Unknown");
  const confidence = result.confidence ?? 0;
  const concernLevel = result.concern_level ?? "unknown";
  const nextSteps = result.next_steps ?? [];

  // Map confidence to a simple text rating
  const confidenceRating = confidence > 0.85 ? "High" : confidence > 0.6 ? "Moderate" : "Low";

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Main Insight */}
      <div className="space-y-3">
        <h3 className="text-[11px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em]">
          Primary Finding
        </h3>
        <div className="flex items-center justify-between p-5 bg-[#F8FAFC] border border-[#E2E8F0] rounded-2xl shadow-sm">
          <div className="space-y-1.5">
            <p className="font-display text-2xl font-medium text-[#0F172A] tracking-tight">{label}</p>
            <p className="text-[13px] text-[#64748B]">
              Confidence level: <span className="font-semibold text-[#0F172A]">{confidenceRating}</span> ({(confidence * 100).toFixed(1)}%)
            </p>
          </div>
          <StatusBadge level={concernLevel} />
        </div>
      </div>

      {/* Top 3 Patterns (Progress Bars) */}
      {result.top_3 && result.top_3.length > 0 && (
        <div className="space-y-4 pt-2">
          <h3 className="text-[11px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em]">
            Pattern matches
          </h3>
          <div className="space-y-3.5">
            {result.top_3.map((item, idx) => {
              const perc = (item.probability * 100).toFixed(1);
              return (
                <div key={item.label} className="space-y-1.5">
                  <div className="flex justify-between items-center text-[13px]">
                    <span className={clsx("font-medium", idx === 0 ? "text-[#0F172A]" : "text-[#475569]")}>
                      {CLASS_LABELS[item.label] ?? item.label}
                    </span>
                    <span className={clsx("font-medium", idx === 0 ? "text-[#0B7FEA]" : "text-[#64748B]")}>
                      {perc}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-[#F1F5F9] rounded-full overflow-hidden">
                    <div 
                      className={clsx("h-full rounded-full transition-all duration-1000 ease-out", idx === 0 ? "bg-[#0B7FEA]" : "bg-[#CBD5E1]")} 
                      style={{ width: `${perc}%` }} 
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Next steps */}
      {nextSteps.length > 0 && (
        <div className="space-y-4 pt-2">
          <h3 className="text-[11px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em]">
            Recommended Next Steps
          </h3>
          <div className="space-y-3 bg-white border border-[#E2E8F0] rounded-2xl p-5 shadow-sm">
            {nextSteps.map((step, i) => (
              <div key={i} className="flex items-start gap-3">
                <CheckCircle2 size={18} className="text-[#10B981] mt-0.5 flex-shrink-0" />
                <p className="text-[14px] text-[#0F172A] leading-relaxed">{step}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="flex items-start gap-3 px-5 py-4 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl mt-4">
        <Info size={16} className="text-[#64748B] flex-shrink-0 mt-0.5" />
        <p className="text-[12px] text-[#475569] leading-relaxed">
          <strong>Educational insight only.</strong> This analysis is generated by an AI model and should not be used as a definitive medical diagnosis. Always consult a dermatologist for skin concerns.
        </p>
      </div>
    </div>
  );
}
