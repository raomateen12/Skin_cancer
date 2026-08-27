"use client";

import { Activity, WifiOff, Clock, ChevronRight, AlertTriangle, Info, CheckCircle2, AlertOctagon, Layers } from "lucide-react";
import StatusBadge from "@/components/shared/StatusBadge";
import clsx from "clsx";

export interface PredictResult {
  ok?: boolean;
  available?: boolean;  // optional — normalized in api.ts
  // Rejection fields
  rejected?: boolean;
  rejection_reason?: string;
  guidance?: string;
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
  gradcam_images?: { original: string; gradcam: string; eigencam: string } | null;
  gradcam_images_list?: string[];
  xai_error?: string | null;
  // Segmentation
  segmentation_available?: boolean;
  segmentation_overlay?: string | null;
  segmentation_mask?: string | null;
  segmentation_heatmap?: string | null;
  lesion_morphology?: {
    lesion_detected?: boolean;
    area_pct?: number;
    perimeter_px?: number;
    compactness?: number;
    solidity?: number;
    border_irregularity_score?: number;
    aspect_ratio?: number;
    centroid?: { x: number; y: number };
    bounding_box?: { x: number; y: number; width: number; height: number };
    num_lesion_components?: number;
  } | null;
  seg_error?: string | null;
  // Counterfactual ABCD Explanations
  counterfactuals_available?: boolean;
  counterfactuals?: import("@/lib/api").CounterfactualsMap | null;
  cf_error?: string | null;
  // Multimodal Patient Metadata Fusion
  metadata_fusion_available?: boolean;
  fusion_predicted_code?: string | null;
  fusion_predicted_name?: string | null;
  fusion_confidence?: number | null;
  fusion_agrees_with_image_only?: boolean | null;
  fusion_disagreement_note?: string | null;
  fusion_error?: string | null;
  image_quality_warning?: string | null;
  // Clinical Alert System
  alert_level?: "high_risk" | "low_confidence" | "normal";
  alert_message?: string | null;
  skin_tone_reliability_note?: string | null;
  ita_group?: string | null;
  ita_value?: number | null;
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

  // ── Rejected image state ────────────────────────────────────────────────
  if (result.rejected === true) {
    return (
      <div className="space-y-4">
        <div className="p-8 bg-[#FFFBEB] border border-[#FEF08A] rounded-[1.25rem] flex flex-col items-center gap-4 text-center">
          <div className="w-12 h-12 rounded-xl bg-white border border-[#FEF08A] shadow-sm flex items-center justify-center">
            <AlertTriangle size={20} className="text-[#F59E0B]" />
          </div>
          <div className="space-y-2">
            <p className="font-display text-[15px] font-semibold text-[#92400E]">
              Image not suitable for analysis
            </p>
            <p className="text-[13px] text-[#78350F] leading-relaxed max-w-[340px] mx-auto">
              {result.rejection_reason ?? "This image does not appear to be a close-up skin lesion photo."}
            </p>
          </div>
          {result.guidance && (
            <div className="flex items-start gap-2.5 px-4 py-3 bg-white border border-[#FDE68A] rounded-xl text-left w-full max-w-sm">
              <Info size={14} className="text-[#F59E0B] flex-shrink-0 mt-0.5" />
              <p className="text-[12px] text-[#92400E] leading-relaxed">{result.guidance}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Show error/unavailable state only when the backend explicitly signals failure:
  // ok===false AND (backend is down OR checkpoint is missing).
  // If available is undefined but prediction fields exist, fall through to results.
  const hasPrediction =
    Boolean(result.predicted_code) ||
    Boolean(result.predicted_name) ||
    Boolean(result.predicted_class) ||
    Boolean(result.predicted_label);

  const isFailure =
    result.ok === false ||
    (result.available === false && !hasPrediction);

  if (isFailure) {
    const isBackendDown = result.error?.includes("Backend is not running") || result.error?.includes("not running");
    const isCheckpointMissing = Boolean(result.missing_path) || result.error?.includes("weights") || result.error?.includes("checkpoint");

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

  // Full result — read predicted_name first (HF backend), fall back to predicted_label / predicted_class lookup
  const label =
    result.predicted_name ??
    result.predicted_label ??
    (result.predicted_code ? CLASS_LABELS[result.predicted_code] ?? result.predicted_code : null) ??
    (result.predicted_class ? CLASS_LABELS[result.predicted_class] ?? result.predicted_class : "Unknown");
  const confidence = result.confidence ?? 0;
  const concernLevel = result.concern_level ?? "unknown";
  const nextSteps = result.next_steps ?? [];

  // Map confidence to a simple text rating
  const confidenceRating = confidence > 0.85 ? "High" : confidence > 0.6 ? "Moderate" : "Low";

  return (
    <div className="space-y-6 animate-fade-in">
      {/* ── Clinical Alert Banner (High Risk) ── */}
      {result.alert_level === "high_risk" && (
        <div className="flex items-start gap-3.5 p-4.5 bg-[#FEF2F2] border border-[#FECACA] rounded-2xl shadow-sm">
          <div className="w-8 h-8 rounded-xl bg-white border border-[#FECACA] shadow-sm flex items-center justify-center flex-shrink-0 mt-0.5">
            <AlertOctagon size={18} className="text-[#DC2626]" />
          </div>
          <div className="space-y-1">
            <p className="text-[13px] font-semibold text-[#991B1B]">
              High-Risk Lesion Category Alert
            </p>
            <p className="text-[12px] text-[#B91C1C] leading-relaxed">
              {result.alert_message ?? "This prediction falls into a higher-risk lesion category. Please consult a dermatologist promptly for professional evaluation."}
            </p>
          </div>
        </div>
      )}

      {/* ── Clinical Alert Banner (Low Confidence) ── */}
      {result.alert_level === "low_confidence" && (
        <div className="flex items-start gap-3.5 p-4.5 bg-[#FFFBEB] border border-[#FDE68A] rounded-2xl shadow-sm">
          <div className="w-8 h-8 rounded-xl bg-white border border-[#FDE68A] shadow-sm flex items-center justify-center flex-shrink-0 mt-0.5">
            <AlertTriangle size={18} className="text-[#D97706]" />
          </div>
          <div className="space-y-1">
            <p className="text-[13px] font-semibold text-[#92400E]">
              Low Confidence Prediction Note
            </p>
            <p className="text-[12px] text-[#B45309] leading-relaxed">
              {result.alert_message ?? "The model's confidence in this prediction is relatively low. We recommend consulting a dermatologist for a definitive diagnosis."}
            </p>
          </div>
        </div>
      )}

      {/* ── Skin Tone Reliability Note (Informational Context) ── */}
      {result.skin_tone_reliability_note && (
        <div className="flex items-start gap-3 px-4 py-3 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl shadow-sm">
          <Info size={15} className="text-[#0B7FEA] flex-shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <p className="text-[11px] font-semibold text-[#0F172A] uppercase tracking-wider">
              Skin-Tone Reliability Context {result.ita_group ? `(${result.ita_group.toUpperCase()})` : ""}
            </p>
            <p className="text-[12px] text-[#475569] leading-relaxed">
              {result.skin_tone_reliability_note}
            </p>
          </div>
        </div>
      )}

      {/* Main Insight */}
      <div className="space-y-3">
        <h3 className="text-[11px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em]">
          Primary Finding
        </h3>
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 sm:p-5 bg-[#F8FAFC] border border-[#E2E8F0] rounded-2xl shadow-sm">
          <div className="space-y-1.5 w-full">
            <p className="font-display text-xl sm:text-2xl font-medium text-[#0F172A] tracking-tight break-words pr-2">{label}</p>
            <p className="text-[12px] sm:text-[13px] text-[#64748B]">
              Confidence level: <span className="font-semibold text-[#0F172A]">{confidenceRating}</span> ({(confidence * 100).toFixed(1)}%)
            </p>
          </div>
          <div className="flex-shrink-0">
            <StatusBadge level={concernLevel} />
          </div>
        </div>
      </div>

      {/* ── Multimodal Patient Metadata Fusion Card ── */}
      {result.metadata_fusion_available && result.fusion_predicted_name && (
        <div className="space-y-3 pt-1">
          <div className="flex items-center justify-between">
            <h3 className="text-[11px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em] flex items-center gap-1.5">
              <Layers size={13} className="text-[#0B7FEA]" />
              Multimodal Analysis (Image + Metadata)
            </h3>
            {result.fusion_agrees_with_image_only ? (
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-[#F0FDF4] text-[#15803D] border border-[#BBF7D0]">
                <CheckCircle2 size={11} className="text-[#16A34A]" />
                Agrees with Image Model
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-[#FFFBEB] text-[#B45309] border border-[#FDE68A]">
                <AlertTriangle size={11} className="text-[#D97706]" />
                Multimodal Disagreement
              </span>
            )}
          </div>

          <div className="p-4 sm:p-5 bg-white border border-[#E2E8F0] rounded-2xl shadow-sm space-y-3">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-[11px] text-[#64748B] uppercase tracking-wider font-medium">
                  Multimodal Prediction
                </p>
                <p className="font-display text-[17px] font-semibold text-[#0F172A] mt-0.5">
                  {result.fusion_predicted_name}
                </p>
              </div>
              <div className="text-right">
                <p className="text-[11px] text-[#64748B] uppercase tracking-wider font-medium">
                  Multimodal Conf.
                </p>
                <p className="font-display text-[17px] font-semibold text-[#0B7FEA] mt-0.5">
                  {result.fusion_confidence !== undefined && result.fusion_confidence !== null
                    ? `${(result.fusion_confidence * 100).toFixed(1)}%`
                    : "N/A"}
                </p>
              </div>
            </div>

            {/* Disagreement Callout */}
            {!result.fusion_agrees_with_image_only && result.fusion_disagreement_note && (
              <div className="p-3.5 bg-[#FFFBEB] border border-[#FDE68A] rounded-xl flex items-start gap-2.5 text-[12px] text-[#92400E] leading-relaxed">
                <AlertTriangle size={15} className="text-[#D97706] flex-shrink-0 mt-0.5" />
                <p>{result.fusion_disagreement_note}</p>
              </div>
            )}

            {/* Agreement Note */}
            {result.fusion_agrees_with_image_only && (
              <div className="p-3 bg-[#F0FDF4] border border-[#DCFCE7] rounded-xl flex items-start gap-2 text-[12px] text-[#166534] leading-relaxed">
                <CheckCircle2 size={14} className="text-[#16A34A] flex-shrink-0 mt-0.5" />
                <p>
                  Both the visual EfficientNet-B0 backbone and patient demographic metadata independently support <strong>{result.fusion_predicted_name}</strong>.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Top 3 Patterns (Progress Bars) — handles both backend shapes */}
      {(() => {
        // Normalise to a single array regardless of which shape the backend returned
        const items: Array<{ displayName: string; pct: number }> =
          result.top_3 && result.top_3.length > 0
            ? result.top_3.map((i) => ({
                displayName: CLASS_LABELS[i.label] ?? i.label,
                pct: i.probability * 100,
              }))
            : result.top_predictions && result.top_predictions.length > 0
            ? result.top_predictions.map((i) => ({
                displayName: i.name ?? CLASS_LABELS[i.code] ?? i.code,
                pct: i.confidence * 100,
              }))
            : [];

        if (items.length === 0) return null;

        return (
          <div className="space-y-4 pt-2">
            <h3 className="text-[11px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em]">
              Pattern matches
            </h3>
            <div className="space-y-3.5">
              {items.map((item, idx) => {
                const perc = item.pct.toFixed(1);
                return (
                  <div key={idx} className="space-y-1.5">
                    <div className="flex justify-between items-center text-[13px]">
                      <span className={clsx("font-medium", idx === 0 ? "text-[#0F172A]" : "text-[#475569]")}>
                        {item.displayName}
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
        );
      })()}

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

      {/* Image quality warning */}
      {result.image_quality_warning && (
        <div className="flex items-start gap-3 px-5 py-4 bg-[#FFFBEB] border border-[#FEF08A] rounded-xl">
          <AlertTriangle size={14} className="text-[#F59E0B] flex-shrink-0 mt-0.5" />
          <p className="text-[12px] text-[#92400E] leading-relaxed">
            <strong>Image quality note.</strong> {result.image_quality_warning}
          </p>
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
