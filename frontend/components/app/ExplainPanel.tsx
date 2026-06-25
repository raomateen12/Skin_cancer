"use client";

import { Eye, Info, CheckSquare, ShieldCheck, Image as ImageIcon } from "lucide-react";
import { type PredictResult } from "@/components/app/ResultPanel";
import StatusBadge from "@/components/shared/StatusBadge";

interface ExplainPanelProps {
  result?: PredictResult | null;
  uploadedImageUrl?: string | null;
  gradcamImages?: string[];
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

const CHECKLIST = [
  "Is the image clear, in-focus, and well-lit?",
  "Is the lesion centered and filling most of the frame?",
  "Does the highlighted region overlap the visible lesion area?",
  "Are there visible changes in size, border, color, or texture?",
  "Would a dermatologist review be appropriate for this lesion?",
];

export default function ExplainPanel({
  result,
  uploadedImageUrl,
  gradcamImages = [],
}: ExplainPanelProps) {
  const hasResult = result && (result.available || result.ok);
  const hasGradcam = gradcamImages.length > 0;
  const label =
    result?.predicted_name ??
    result?.predicted_label ??
    (result?.predicted_class ? CLASS_LABELS[result.predicted_class] ?? result.predicted_class : null);
  const confidence = result?.confidence;
  const concernLevel = result?.concern_level;

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h2 className="font-display text-2xl font-semibold text-[#0F172A]">
          Visual Explanation
        </h2>
        <p className="text-[14px] text-[#64748B] mt-1.5">
          Understand what the model reviewed before showing an AI-assisted insight.
        </p>
      </div>

      {/* Result Recap Card */}
      <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] p-6 shadow-soft">
        <div className="flex items-center gap-2 mb-4">
          <Eye size={16} className="text-[#0B7FEA]" />
          <h3 className="font-display text-[15px] font-medium text-[#0F172A]">
            Analysis Recap
          </h3>
        </div>
        {hasResult && label ? (
          <div className="grid sm:grid-cols-3 gap-6">
            <div className="space-y-1">
              <p className="text-[10px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em]">
                Primary Finding
              </p>
              <p className="font-display text-[16px] font-medium text-[#0F172A]">{label}</p>
            </div>
            {confidence !== undefined && (
              <div className="space-y-1">
                <p className="text-[10px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em]">
                  Confidence
                </p>
                <p className="font-display text-[16px] font-medium text-[#0F172A]">
                  {(confidence * 100).toFixed(1)}%
                </p>
              </div>
            )}
            {concernLevel && (
              <div className="space-y-1">
                <p className="text-[10px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em]">
                  Concern Level
                </p>
                <StatusBadge level={concernLevel} />
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-4 py-3">
            <div className="w-10 h-10 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] flex items-center justify-center">
              <Eye size={20} className="text-[#94A3B8]" />
            </div>
            <div>
              <p className="text-[14px] font-medium text-[#0F172A]">No analysis yet</p>
              <p className="text-[13px] text-[#64748B] mt-0.5">
                Analyze an image first to see a visual explanation here.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Visual Explanation Grid */}
      <div>
        <h3 className="font-display text-[16px] font-medium text-[#0F172A] mb-4">
          Visual attention maps
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          {/* Original Image */}
          <div className="space-y-3">
            <p className="text-[11px] font-semibold text-[#64748B] uppercase tracking-[0.15em]">
              Original image
            </p>
            <div className="bg-white border border-[#E2E8F0] rounded-2xl overflow-hidden shadow-soft aspect-square flex items-center justify-center p-2">
              {uploadedImageUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={uploadedImageUrl}
                  alt="Uploaded skin lesion"
                  className="w-full h-full object-cover rounded-[0.85rem]"
                />
              ) : (
                <div className="flex flex-col items-center gap-3 text-center p-4">
                  <div className="w-12 h-12 rounded-[14px] bg-[#F8FAFC] border border-[#E2E8F0] flex items-center justify-center">
                    <ImageIcon size={22} className="text-[#94A3B8]" />
                  </div>
                  <p className="text-[12px] font-medium text-[#64748B] leading-relaxed">
                    No image uploaded<br/>in this session
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Grad-CAM */}
          <div className="space-y-3">
            <p className="text-[11px] font-semibold text-[#64748B] uppercase tracking-[0.15em]">
              Grad-CAM focus
            </p>
            <div className="bg-white border border-[#E2E8F0] rounded-2xl overflow-hidden shadow-soft aspect-square flex items-center justify-center p-2">
              {hasGradcam && gradcamImages[1] ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={gradcamImages[1]}
                  alt="Grad-CAM attention map"
                  className="w-full h-full object-cover rounded-[0.85rem]"
                />
              ) : (
                <div className="flex flex-col items-center gap-3 text-center p-4">
                  <div className="w-12 h-12 rounded-[14px] bg-[#F8FAFC] border border-[#E2E8F0] flex items-center justify-center">
                    <Eye size={22} className="text-[#94A3B8]" />
                  </div>
                  <div>
                    <p className="text-[12px] font-medium text-[#475569]">
                      Attention map not available
                    </p>
                    <p className="text-[11px] text-[#94A3B8] mt-1.5 leading-relaxed">
                      This appears when visual explanation generation is connected.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* EigenCAM */}
          <div className="space-y-3">
            <p className="text-[11px] font-semibold text-[#64748B] uppercase tracking-[0.15em]">
              EigenCAM overlay
            </p>
            <div className="bg-white border border-[#E2E8F0] rounded-2xl overflow-hidden shadow-soft aspect-square flex items-center justify-center p-2">
              {hasGradcam && gradcamImages[2] ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={gradcamImages[2]}
                  alt="EigenCAM attention map"
                  className="w-full h-full object-cover rounded-[0.85rem]"
                />
              ) : (
                <div className="flex flex-col items-center gap-3 text-center p-4">
                  <div className="w-12 h-12 rounded-[14px] bg-[#F8FAFC] border border-[#E2E8F0] flex items-center justify-center">
                    <Eye size={22} className="text-[#94A3B8]" />
                  </div>
                  <div>
                    <p className="text-[12px] font-medium text-[#475569]">
                      Attention map not available
                    </p>
                    <p className="text-[11px] text-[#94A3B8] mt-1.5 leading-relaxed">
                      This appears when visual explanation generation is connected.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom section: 2 columns */}
      <div className="grid md:grid-cols-2 gap-6">

        {/* How to read this */}
        <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] p-6 shadow-soft">
          <div className="flex items-center gap-2.5 mb-4">
            <Info size={16} className="text-[#0B7FEA]" />
            <h4 className="font-display text-[15px] font-medium text-[#0F172A]">
              How to read this
            </h4>
          </div>
          <ul className="space-y-3.5">
            {[
              "Highlighted areas show regions that influenced the model's prediction.",
              "The highlighted region should ideally overlap the lesion area.",
              "Heatmaps are transparency aids, not clinical proof.",
              "A dermatologist should always make the final clinical decision.",
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-3">
                <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-[#0B7FEA] flex-shrink-0" />
                <p className="text-[13px] text-[#475569] leading-relaxed">{item}</p>
              </li>
            ))}
          </ul>
        </div>

        {/* Review checklist */}
        <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] p-6 shadow-soft">
          <div className="flex items-center gap-2.5 mb-4">
            <CheckSquare size={16} className="text-[#64748B]" />
            <h4 className="font-display text-[15px] font-medium text-[#0F172A]">
              Review checklist
            </h4>
          </div>
          <ul className="space-y-3.5">
            {CHECKLIST.map((item, i) => (
              <li key={i} className="flex items-start gap-3">
                <div className="mt-0.5 w-4 h-4 rounded-sm flex-shrink-0 border border-[#E2E8F0] bg-[#F8FAFC]" />
                <p className="text-[13px] text-[#475569] leading-relaxed">{item}</p>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Safety Note */}
      <div className="flex items-start gap-3 px-5 py-4 bg-[#F8FAFC] border border-[#E2E8F0] rounded-[1.25rem]">
        <ShieldCheck size={18} className="text-[#64748B] flex-shrink-0 mt-0.5" />
        <p className="text-[13px] text-[#475569] leading-relaxed">
          <span className="font-medium text-[#0F172A]">Safety note.</span>{" "}
          Visual explanations are designed to support transparency. They do not prove a diagnosis and
          should not replace professional medical evaluation by a qualified dermatologist.
        </p>
      </div>
    </div>
  );
}
