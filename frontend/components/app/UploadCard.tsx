"use client";

import { useState, useCallback, useRef } from "react";
import { Upload, Image as ImageIcon, X, AlertCircle, User, Activity, ChevronDown, ChevronUp } from "lucide-react";
import clsx from "clsx";
import { predictImage } from "@/lib/api";

interface UploadCardProps {
  onResult: (result: unknown) => void;
  onAnalyzing: (loading: boolean) => void;
  onImageSelected?: (url: string | null) => void;
  onFileSelected?: (file: File | null) => void;
}

const LOCALIZATIONS = [
  { value: "back", label: "Back" },
  { value: "lower extremity", label: "Lower Extremity (Legs, Hips)" },
  { value: "trunk", label: "Trunk" },
  { value: "upper extremity", label: "Upper Extremity (Arms, Shoulders)" },
  { value: "abdomen", label: "Abdomen" },
  { value: "face", label: "Face" },
  { value: "chest", label: "Chest" },
  { value: "unknown", label: "Unknown / Other" },
];

export default function UploadCard({ onResult, onAnalyzing, onImageSelected, onFileSelected }: UploadCardProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Patient metadata state (Optional)
  const [age, setAge] = useState<string>("");
  const [sex, setSex] = useState<string>("");
  const [localization, setLocalization] = useState<string>("");
  const [metaOpen, setMetaOpen] = useState<boolean>(true);

  const handleFile = (f: File) => {
    setError(null);
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file (PNG, JPG, JPEG, or WEBP).");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("File size must be 10 MB or less.");
      return;
    }
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
    onImageSelected?.(url);
    onFileSelected?.(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  };

  const onDragLeave = () => setDragging(false);

  const handleAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    onAnalyzing(true);
    onResult(null); // clear stale previous result immediately
    try {
      const parsedAge = age.trim() !== "" ? parseFloat(age) : null;
      const metadata = {
        patientAge: parsedAge !== null && !isNaN(parsedAge) ? parsedAge : null,
        patientSex: sex.trim() !== "" ? sex : null,
        patientLocalization: localization.trim() !== "" ? localization : null,
      };
      const result = await predictImage(file, false, metadata);
      onResult(result);
    } catch {
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setAnalyzing(false);
      onAnalyzing(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    onImageSelected?.(null);
    onFileSelected?.(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="space-y-4">
      {/* Upload zone */}
      <div
        id="upload-dropzone"
        className={clsx(
          "relative border-2 border-dashed rounded-[1.25rem] p-6 transition-all duration-300 cursor-pointer flex flex-col items-center justify-center min-h-[220px]",
          dragging
            ? "border-[#0B7FEA] bg-[#F0F7FF] shadow-inner"
            : preview
            ? "border-transparent bg-white shadow-soft"
            : "border-[#CBD5E1] bg-white hover:border-[#94A3B8] hover:bg-[#F8FAFC]"
        )}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !preview && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg,image/webp"
          className="hidden"
          id="file-input"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />

        {preview ? (
          <div className="relative w-full h-full flex flex-col items-center">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preview}
              alt="Uploaded skin lesion preview"
              className="w-full h-40 sm:h-48 object-contain rounded-xl bg-[#F8FAFC] border border-[#E2E8F0]"
            />
            <button
              onClick={(e) => { e.stopPropagation(); clearFile(); }}
              className="absolute top-2 right-2 w-8 h-8 bg-white/90 backdrop-blur-sm border border-[#E2E8F0] rounded-full flex items-center justify-center shadow-sm hover:bg-white transition-colors"
              aria-label="Remove image"
            >
              <X size={16} className="text-[#64748B]" />
            </button>
            <div className="mt-5 flex items-center justify-between w-full px-2 border-t border-[#E2E8F0] pt-3">
              <span className="text-[13px] font-medium text-[#0F172A] truncate max-w-[140px] sm:max-w-[200px]">
                {file?.name}
              </span>
              <span className="text-[10px] sm:text-[11px] font-medium tracking-wide uppercase text-[#10B981]">Ready to analyze</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 py-4">
            <div className={clsx(
              "w-14 h-14 rounded-2xl flex items-center justify-center transition-all duration-300",
              dragging ? "bg-[#0B7FEA] shadow-md scale-105" : "bg-[#F8FAFC] border border-[#E2E8F0] shadow-sm"
            )}>
              {dragging ? (
                <ImageIcon size={24} className="text-white" />
              ) : (
                <Upload size={24} className="text-[#64748B]" />
              )}
            </div>
            <div className="text-center">
              <p className="text-[15px] font-medium text-[#0F172A]">
                {dragging ? "Drop image to upload" : "Click or drag image to upload"}
              </p>
              <p className="text-[13px] text-[#64748B] mt-1.5 leading-relaxed">PNG, JPG up to 10MB</p>
            </div>
          </div>
        )}
      </div>

      {/* Optional Patient Clinical Metadata Context */}
      <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] p-4 shadow-soft">
        <button
          type="button"
          onClick={() => setMetaOpen(!metaOpen)}
          className="w-full flex items-center justify-between text-left focus:outline-none"
        >
          <div className="flex items-center gap-2">
            <User size={15} className="text-[#0B7FEA]" />
            <span className="text-[13px] font-semibold text-[#0F172A]">
              Patient Context & Metadata
            </span>
            <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-[#F1F5F9] text-[#64748B] border border-[#E2E8F0]">
              Optional
            </span>
          </div>
          <div className="text-[#94A3B8]">
            {metaOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>
        </button>

        {metaOpen && (
          <div className="mt-3.5 pt-3 border-t border-[#F1F5F9] space-y-3 animate-fade-in text-[13px]">
            <p className="text-[12px] text-[#64748B] leading-relaxed">
              Optionally provide patient demographics to enable multimodal fusion prediction.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {/* Age input */}
              <div>
                <label htmlFor="patient-age" className="block text-[11px] font-semibold text-[#475569] uppercase tracking-wider mb-1">
                  Age (Years)
                </label>
                <input
                  id="patient-age"
                  type="number"
                  min="0"
                  max="85"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  placeholder="e.g. 45"
                  className="w-full px-3 py-2 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-[13px] text-[#0F172A] focus:outline-none focus:border-[#0B7FEA] focus:bg-white transition-colors"
                />
              </div>

              {/* Sex select */}
              <div>
                <label htmlFor="patient-sex" className="block text-[11px] font-semibold text-[#475569] uppercase tracking-wider mb-1">
                  Biological Sex
                </label>
                <select
                  id="patient-sex"
                  value={sex}
                  onChange={(e) => setSex(e.target.value)}
                  className="w-full px-3 py-2 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-[13px] text-[#0F172A] focus:outline-none focus:border-[#0B7FEA] focus:bg-white transition-colors"
                >
                  <option value="">Prefer not to say</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>

              {/* Localization select */}
              <div>
                <label htmlFor="patient-loc" className="block text-[11px] font-semibold text-[#475569] uppercase tracking-wider mb-1">
                  Lesion Location
                </label>
                <select
                  id="patient-loc"
                  value={localization}
                  onChange={(e) => setLocalization(e.target.value)}
                  className="w-full px-3 py-2 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-[13px] text-[#0F172A] focus:outline-none focus:border-[#0B7FEA] focus:bg-white transition-colors"
                >
                  <option value="">Select location</option>
                  {LOCALIZATIONS.map((loc) => (
                    <option key={loc.value} value={loc.value}>
                      {loc.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2.5 px-4 py-3 bg-[#FEF2F2] border border-[#FECACA] rounded-xl animate-fade-in shadow-sm">
          <AlertCircle size={16} className="text-[#EF4444] flex-shrink-0" />
          <p className="text-[13px] font-medium text-[#DC2626] leading-relaxed">{error}</p>
        </div>
      )}

      {/* Analyze button */}
      <button
        id="analyze-button"
        onClick={handleAnalyze}
        disabled={!file || analyzing}
        className={clsx(
          "w-full py-3.5 rounded-xl text-[14px] font-medium transition-all duration-300 flex items-center justify-center gap-2.5",
          file && !analyzing
            ? "bg-[#0B7FEA] hover:bg-[#0ea5e9] text-white shadow-soft-lg hover:-translate-y-[1px]"
            : "bg-[#F1F5F9] text-[#94A3B8] cursor-not-allowed"
        )}
      >
        {analyzing ? (
          <>
            <div className="w-4 h-4 border-[2px] border-white/30 border-t-white rounded-full animate-spin" />
            Analyzing image & metadata...
          </>
        ) : (
          "Analyze Image"
        )}
      </button>
    </div>
  );
}

