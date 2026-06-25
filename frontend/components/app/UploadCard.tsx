"use client";

import { useState, useCallback, useRef } from "react";
import { Upload, Image as ImageIcon, X, AlertCircle } from "lucide-react";
import clsx from "clsx";
import { predictImage } from "@/lib/api";

interface UploadCardProps {
  onResult: (result: unknown) => void;
  onAnalyzing: (loading: boolean) => void;
  onImageSelected?: (url: string | null) => void;
}

export default function UploadCard({ onResult, onAnalyzing, onImageSelected }: UploadCardProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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
      const result = await predictImage(file);
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
              className="w-full h-48 object-contain rounded-xl bg-[#F8FAFC] border border-[#E2E8F0]"
            />
            <button
              onClick={(e) => { e.stopPropagation(); clearFile(); }}
              className="absolute top-2 right-2 w-8 h-8 bg-white/90 backdrop-blur-sm border border-[#E2E8F0] rounded-full flex items-center justify-center shadow-sm hover:bg-white transition-colors"
              aria-label="Remove image"
            >
              <X size={16} className="text-[#64748B]" />
            </button>
            <div className="mt-5 flex items-center justify-between w-full px-2 border-t border-[#E2E8F0] pt-3">
              <span className="text-[13px] font-medium text-[#0F172A] truncate max-w-[200px]">
                {file?.name}
              </span>
              <span className="text-[11px] font-medium tracking-wide uppercase text-[#10B981]">Ready to analyze</span>
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
            Analyzing image...
          </>
        ) : (
          "Analyze Image"
        )}
      </button>
    </div>
  );
}
