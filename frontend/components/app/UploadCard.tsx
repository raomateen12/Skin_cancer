"use client";

import { useState, useCallback, useRef } from "react";
import {
  Upload,
  Image as ImageIcon,
  X,
  AlertCircle,
  User,
  ChevronDown,
  ChevronUp,
  Check,
} from "lucide-react";
import clsx from "clsx";
import { predictImage } from "@/lib/api";

interface UploadCardProps {
  onResult: (result: unknown) => void;
  onAnalyzing: (loading: boolean) => void;
  onImageSelected?: (url: string | null) => void;
  onFileSelected?: (file: File | null) => void;
}

const SAMPLE_IMAGES = [
  {
    id: "melanoma",
    label: "Melanoma",
    src: "/samples/sample_melanoma.jpg",
    fileName: "sample_melanoma.jpg",
    age: 75, sex: "male", localization: "abdomen",
  },
  {
    id: "nevus",
    label: "Nevus",
    src: "/samples/sample_nevus.jpg",
    fileName: "sample_nevus.jpg",
    age: 55, sex: "male", localization: "abdomen",
  },
  {
    id: "bcc",
    label: "Basal Cell Ca.",
    src: "/samples/sample_bcc.jpg",
    fileName: "sample_bcc.jpg",
    age: 85, sex: "female", localization: "back",
  },
  {
    id: "bkl",
    label: "Keratosis",
    src: "/samples/sample_bkl.jpg",
    fileName: "sample_bkl.jpg",
    age: 75, sex: "male", localization: "back",
  },
  {
    id: "vascular",
    label: "Vascular",
    src: "/samples/sample_vascular.jpg",
    fileName: "sample_vascular.jpg",
    age: 55, sex: "male", localization: "lower extremity",
  },
];

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
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [loadingSampleId, setLoadingSampleId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const [age, setAge] = useState<string>("");
  const [sex, setSex] = useState<string>("");
  const [localization, setLocalization] = useState<string>("");
  const [metaOpen, setMetaOpen] = useState<boolean>(false);

  const applyFile = (f: File, url: string) => {
    setFile(f);
    setPreview(url);
    onImageSelected?.(url);
    onFileSelected?.(f);
  };

  const handleFile = (f: File) => {
    setError(null);
    setSelectedSampleId(null);
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file (PNG, JPG, or WEBP).");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("File size must be 10 MB or less.");
      return;
    }
    applyFile(f, URL.createObjectURL(f));
  };

  const handleSampleClick = async (sample: typeof SAMPLE_IMAGES[0]) => {
    if (loadingSampleId) return;
    setError(null);
    setLoadingSampleId(sample.id);
    try {
      const res = await fetch(sample.src);
      if (!res.ok) throw new Error("fetch failed");
      const blob = await res.blob();
      const sampleFile = new File([blob], sample.fileName, { type: "image/jpeg" });
      setSelectedSampleId(sample.id);
      applyFile(sampleFile, sample.src);
      setAge(sample.age.toString());
      setSex(sample.sex);
      setLocalization(sample.localization);
      setMetaOpen(true);
    } catch {
      setError("Could not load sample. Please try again.");
    } finally {
      setLoadingSampleId(null);
    }
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    onAnalyzing(true);
    onResult(null);
    try {
      const parsedAge = age.trim() ? parseFloat(age) : null;
      const result = await predictImage(file, false, {
        patientAge: parsedAge && !isNaN(parsedAge) ? parsedAge : null,
        patientSex: sex.trim() || null,
        patientLocalization: localization.trim() || null,
      });
      onResult(result);
    } catch {
      setError("Analysis failed. Please make sure the backend is running.");
    } finally {
      setAnalyzing(false);
      onAnalyzing(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setSelectedSampleId(null);
    setError(null);
    setAge(""); setSex(""); setLocalization("");
    onImageSelected?.(null);
    onFileSelected?.(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="space-y-4">
      {/* ── Upload Zone ── */}
      <div
        id="upload-dropzone"
        className={clsx(
          "relative border-2 border-dashed rounded-[1.25rem] p-6 transition-all duration-300 cursor-pointer flex flex-col items-center justify-center min-h-[200px]",
          dragging ? "border-[#0B7FEA] bg-[#F0F7FF]"
            : preview ? "border-transparent bg-white shadow-soft"
            : "border-[#CBD5E1] bg-white hover:border-[#0B7FEA] hover:bg-[#F8FAFC]"
        )}
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
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
          <div className="w-full flex flex-col items-center">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preview}
              alt="Selected dermoscopic image"
              className="w-full h-44 object-contain rounded-xl bg-[#F8FAFC] border border-[#E2E8F0]"
            />
            <button
              onClick={(e) => { e.stopPropagation(); clearFile(); }}
              className="absolute top-3 right-3 w-8 h-8 bg-white border border-[#E2E8F0] rounded-full flex items-center justify-center shadow-sm hover:text-red-500 transition-colors cursor-pointer"
            >
              <X size={15} />
            </button>
            <div className="mt-3 w-full flex items-center justify-between text-[12.5px] px-1">
              <span className="text-[#475569] truncate max-w-[180px]">
                {selectedSampleId
                  ? SAMPLE_IMAGES.find(s => s.id === selectedSampleId)?.label + " (sample)"
                  : file?.name}
              </span>
              <span className="text-[#10B981] font-semibold bg-[#ECFDF5] px-2 py-0.5 rounded-full border border-[#A7F3D0]">
                Ready ✓
              </span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-center py-2">
            <div className={clsx(
              "w-12 h-12 rounded-2xl flex items-center justify-center transition-all",
              dragging ? "bg-[#0B7FEA] scale-110" : "bg-[#F1F5F9] border border-[#E2E8F0]"
            )}>
              {dragging
                ? <ImageIcon size={22} className="text-white" />
                : <Upload size={22} className="text-[#0B7FEA]" />}
            </div>
            <div>
              <p className="text-[14px] font-semibold text-[#0F172A]">
                {dragging ? "Drop image here" : "Click or drag your dermoscopic image here"}
              </p>
              <p className="text-[12px] text-[#94A3B8] mt-1">PNG · JPG · WEBP · max 10 MB</p>
            </div>
          </div>
        )}
      </div>

      {/* ── Sample Images Strip ── */}
      <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] p-4 shadow-soft space-y-3">
        <p className="text-[12px] text-[#64748B] font-medium">
          📷 <span className="font-semibold text-[#0F172A]">No dermoscopic image?</span> Click any sample from the HAM10000 dataset:
        </p>
        <div className="grid grid-cols-5 gap-2">
          {SAMPLE_IMAGES.map((sample) => {
            const isSelected = selectedSampleId === sample.id;
            const isLoading = loadingSampleId === sample.id;
            return (
              <button
                key={sample.id}
                type="button"
                onClick={() => handleSampleClick(sample)}
                disabled={!!loadingSampleId}
                className={clsx(
                  "relative flex flex-col items-center gap-1.5 rounded-xl border p-1.5 cursor-pointer transition-all duration-200 group",
                  isSelected
                    ? "border-[#0B7FEA] ring-2 ring-[#0B7FEA]/20 bg-[#F0F7FF]"
                    : "border-[#E2E8F0] hover:border-[#0B7FEA]/50 hover:bg-[#F8FAFC] bg-white"
                )}
              >
                <div className="relative w-full aspect-square rounded-lg overflow-hidden bg-[#F8FAFC]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={sample.src}
                    alt={sample.label}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  {isSelected && (
                    <div className="absolute inset-0 bg-[#0B7FEA]/20 flex items-center justify-center">
                      <div className="w-5 h-5 rounded-full bg-[#0B7FEA] flex items-center justify-center">
                        <Check size={11} strokeWidth={3} className="text-white" />
                      </div>
                    </div>
                  )}
                  {isLoading && (
                    <div className="absolute inset-0 bg-white/70 flex items-center justify-center">
                      <div className="w-4 h-4 border-2 border-[#0B7FEA]/30 border-t-[#0B7FEA] rounded-full animate-spin" />
                    </div>
                  )}
                </div>
                <span className="text-[10px] font-semibold text-[#475569] text-center leading-tight truncate w-full">
                  {sample.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Patient Metadata (collapsible) ── */}
      <div className="bg-white border border-[#E2E8F0] rounded-[1.25rem] p-4 shadow-soft">
        <button
          type="button"
          onClick={() => setMetaOpen(!metaOpen)}
          className="w-full flex items-center justify-between text-left cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <User size={14} className="text-[#0B7FEA]" />
            <span className="text-[13px] font-semibold text-[#0F172A]">Patient Demographics</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[#F1F5F9] text-[#64748B] border border-[#E2E8F0]">
              {selectedSampleId ? "Auto-filled" : "Optional"}
            </span>
          </div>
          {metaOpen ? <ChevronUp size={15} className="text-[#94A3B8]" /> : <ChevronDown size={15} className="text-[#94A3B8]" />}
        </button>

        {metaOpen && (
          <div className="mt-3 pt-3 border-t border-[#F1F5F9] grid grid-cols-1 sm:grid-cols-3 gap-3 text-[13px]">
            <div>
              <label htmlFor="patient-age" className="block text-[11px] font-semibold text-[#475569] uppercase tracking-wider mb-1">Age</label>
              <input
                id="patient-age"
                type="number" min="0" max="100"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                placeholder="e.g. 45"
                className="w-full px-3 py-2 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-[13px] text-[#0F172A] focus:outline-none focus:border-[#0B7FEA] focus:bg-white transition-colors"
              />
            </div>
            <div>
              <label htmlFor="patient-sex" className="block text-[11px] font-semibold text-[#475569] uppercase tracking-wider mb-1">Sex</label>
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
            <div>
              <label htmlFor="patient-loc" className="block text-[11px] font-semibold text-[#475569] uppercase tracking-wider mb-1">Location</label>
              <select
                id="patient-loc"
                value={localization}
                onChange={(e) => setLocalization(e.target.value)}
                className="w-full px-3 py-2 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-[13px] text-[#0F172A] focus:outline-none focus:border-[#0B7FEA] focus:bg-white transition-colors"
              >
                <option value="">Select location</option>
                {LOCALIZATIONS.map((loc) => (
                  <option key={loc.value} value={loc.value}>{loc.label}</option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="flex items-center gap-2.5 px-4 py-3 bg-[#FEF2F2] border border-[#FECACA] rounded-xl">
          <AlertCircle size={15} className="text-[#EF4444] flex-shrink-0" />
          <p className="text-[13px] font-medium text-[#DC2626]">{error}</p>
        </div>
      )}

      {/* ── Analyze Button ── */}
      <button
        id="analyze-button"
        onClick={handleAnalyze}
        disabled={!file || analyzing}
        className={clsx(
          "w-full py-3.5 rounded-xl text-[14px] font-semibold transition-all duration-300 flex items-center justify-center gap-2 cursor-pointer",
          file && !analyzing
            ? "bg-[#0B7FEA] hover:bg-[#0284c7] text-white shadow-soft-lg hover:-translate-y-[1px]"
            : "bg-[#F1F5F9] text-[#94A3B8] cursor-not-allowed"
        )}
      >
        {analyzing ? (
          <>
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Analyzing...
          </>
        ) : file ? (
          "Analyze Image"
        ) : (
          "Upload or Select a Sample to Begin"
        )}
      </button>
    </div>
  );
}
