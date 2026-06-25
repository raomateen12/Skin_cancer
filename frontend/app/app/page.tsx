"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/app/Sidebar";
import UploadCard from "@/components/app/UploadCard";
import ResultPanel, { type PredictResult } from "@/components/app/ResultPanel";
import ExplainPanel from "@/components/app/ExplainPanel";
import AssistantChat from "@/components/app/AssistantChat";
import GuidePanel from "@/components/app/GuidePanel";
import SafetyNote from "@/components/shared/SafetyNote";
import { Info, HelpCircle } from "lucide-react";

type Panel = "analyze" | "result" | "explain" | "ask-assistant" | "guide";

export default function AppPage() {
  const [activePanel, setActivePanel] = useState<Panel>("analyze");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);

  const handleResult = (r: unknown) => {
    setResult(r as PredictResult);
    setActivePanel("result");
  };

  // Clear stale result whenever a new image is chosen
  const handleImageSelected = useCallback((url: string | null) => {
    setUploadedImageUrl(url);
    if (url) {
      // New image picked — wipe any previous result so the panel
      // shows "No analysis yet" instead of a stale error/checkpoint message
      setResult(null);
    }
  }, []);

  return (
    <div className="flex flex-col md:flex-row h-[100dvh] bg-[#F8FAFC] w-full overflow-hidden">
      {/* Sidebar / Bottom Nav */}
      <Sidebar activePanel={activePanel} onPanelChange={(p) => setActivePanel(p as Panel)} />

      {/* Main workspace */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden bg-[#F8FAFC] relative">
        {/* Top bar */}
        <header className="h-14 md:h-16 bg-white flex items-center justify-between px-4 md:px-8 flex-shrink-0 border-b border-[#E2E8F0] z-10 sticky top-0">
          <div className="flex items-center gap-2.5">
            {/* Mobile Logo */}
            <div className="md:hidden w-7 h-7 rounded-[8px] bg-[#0F172A] flex items-center justify-center shadow-sm">
              <span className="text-white text-[10px] font-bold tracking-wider">DL</span>
            </div>
            <h1 className="font-display text-[15px] font-medium text-[#0F172A] capitalize tracking-tight">
              {activePanel.replace("-", " ")}
            </h1>
            <span className="hidden md:inline text-[#CBD5E1] font-light">|</span>
            <span className="hidden md:inline text-[13px] font-medium text-[#64748B]">DermaLens Workspace</span>
          </div>
          <SafetyNote compact />
        </header>

        {/* Panel content */}
        <div
          className={`flex-1 w-full ${
            activePanel === "ask-assistant"
              ? "flex flex-col min-h-0 relative"
              : "overflow-y-auto p-4 sm:p-6 md:p-8 pb-[84px] md:pb-8"
          }`}
        >
          {activePanel === "analyze" && (
            <div className="max-w-5xl mx-auto space-y-8 animate-fade-in">
              <div className="space-y-2">
                <h2 className="font-display text-2xl font-semibold text-[#0F172A] tracking-tight">
                  New Analysis
                </h2>
                <p className="text-[14px] text-[#64748B] leading-relaxed">
                  Upload a dermoscopic image to generate an educational pattern-recognition insight.
                </p>
              </div>

              <div className="grid lg:grid-cols-2 gap-8">
                {/* Left — upload */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-6 h-6 rounded-md bg-[#0F172A] flex items-center justify-center text-white text-[10px] font-bold shadow-sm">
                      1
                    </div>
                    <h3 className="font-display text-[16px] font-medium text-[#0F172A]">
                      Select Image
                    </h3>
                  </div>
                  <UploadCard onResult={handleResult} onAnalyzing={setAnalyzing} onImageSelected={handleImageSelected} />
                </div>

                {/* Right — result preview */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-6 h-6 rounded-md bg-[#F1F5F9] flex items-center justify-center text-[#475569] text-[10px] font-bold border border-[#E2E8F0]">
                      2
                    </div>
                    <h3 className="font-display text-[16px] font-medium text-[#0F172A]">
                      Review Insight
                    </h3>
                  </div>
                  <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] p-6 shadow-soft min-h-[440px]">
                    <ResultPanel result={result} analyzing={analyzing} />
                  </div>
                </div>
              </div>

              {/* Support cards row */}
              <div className="grid md:grid-cols-2 gap-6 pt-6 border-t border-[#E2E8F0]">
                <div className="p-6 bg-white border border-[#E2E8F0] rounded-[1.25rem] shadow-soft card-hover">
                  <div className="flex items-center gap-2.5 mb-4">
                    <Info size={16} className="text-[#0B7FEA]" />
                    <p className="font-display text-[15px] font-medium text-[#0F172A]">Safe imaging practices</p>
                  </div>
                  <ul className="space-y-3">
                    {[
                      "Use bright, diffuse natural light",
                      "Hold camera 15–20 cm away",
                      "Ensure the lesion fills the frame",
                    ].map((item) => (
                      <li key={item} className="flex items-start gap-3 text-[13px] text-[#475569] leading-relaxed">
                        <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-[#E2E8F0] flex-shrink-0" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="p-6 bg-white border border-[#E2E8F0] rounded-[1.25rem] shadow-soft card-hover">
                  <div className="flex items-center gap-2.5 mb-4">
                    <HelpCircle size={16} className="text-[#64748B]" />
                    <p className="font-display text-[15px] font-medium text-[#0F172A]">When to consult a doctor</p>
                  </div>
                  <ul className="space-y-3">
                    {[
                      "Rapidly changing size, shape, or color",
                      "Bleeding, itching, or crusting",
                      "Any concern that persists over time",
                    ].map((item) => (
                      <li key={item} className="flex items-start gap-3 text-[13px] text-[#475569] leading-relaxed">
                        <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-[#E2E8F0] flex-shrink-0" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activePanel === "result" && (
            <div className="max-w-3xl mx-auto animate-fade-in">
              <div className="mb-8">
                <h2 className="font-display text-2xl font-semibold text-[#0F172A] tracking-tight">Clinical Insight</h2>
                <p className="text-[14px] text-[#64748B] mt-2 leading-relaxed">
                  AI-assisted educational interpretation based on your uploaded image.
                </p>
              </div>
              <div className="bg-white border border-[#E2E8F0] rounded-[1.5rem] p-8 shadow-soft-lg">
                <ResultPanel result={result} analyzing={analyzing} />
              </div>
            </div>
          )}

          {activePanel === "explain" && (
            <div className="max-w-5xl mx-auto animate-fade-in">
              <ExplainPanel
                result={result}
                uploadedImageUrl={uploadedImageUrl}
                gradcamImages={
                  result?.gradcam_images && result.gradcam_images.length > 0
                    ? result.gradcam_images
                    : []
                }
              />
            </div>
          )}

          {activePanel === "ask-assistant" && (
            <div className="w-full h-full animate-fade-in flex flex-col min-h-0">
              <AssistantChat />
            </div>
          )}

          {activePanel === "guide" && (
            <div className="max-w-4xl mx-auto animate-fade-in">
              <GuidePanel />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
