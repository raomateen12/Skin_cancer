/**
 * DermaLens AI — Frontend API Client
 * =====================================
 * Connects Next.js frontend to the FastAPI backend.
 * API base URL defaults to http://localhost:8000.
 * Override with NEXT_PUBLIC_API_URL environment variable.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HealthStatus {
  status: string;
  // Documented schema fields
  model_available: boolean;
  checkpoint_path: string;
  class_mapping_available: boolean;
  rag_available: boolean;
  rag_index_path: string;
  message: string;
  // Legacy fields (backward compat with older backend)
  model_loaded?: boolean;
  vectorstore_loaded?: boolean;
  model_name?: string;
}

export interface CounterfactualDetail {
  name: string;
  clinical_code: string;
  description: string;
  original_class: string;
  original_name: string;
  original_confidence: number;
  original_mel_prob: number;
  perturbed_confidence: number;
  perturbed_mel_prob: number;
  mel_prob_delta: number;
  confidence_delta: number;
  new_top_class: string;
  new_top_name: string;
  new_top_confidence: number;
  classification_shifted: boolean;
  area_change_pct: number;
  plain_language_summary: string;
  perturbed_image: string;
  diff_image?: string;
}

export interface CounterfactualsMap {
  border_irregularity?: CounterfactualDetail;
  asymmetry?: CounterfactualDetail;
  diameter?: CounterfactualDetail;
  [key: string]: CounterfactualDetail | undefined;
}

export interface PredictResult {
  ok?: boolean;
  available?: boolean;  // optional — backend may omit this; derived below if absent
  // Rejection fields (when image validation fails)
  rejected?: boolean;
  rejection_reason?: string;
  guidance?: string;
  // Core result fields
  predicted_class?: string;
  predicted_code?: string;
  predicted_label?: string;
  predicted_name?: string;
  confidence?: number;
  concern_level?: "low" | "moderate" | "high";
  concern_message?: string;
  next_steps?: string[];
  // Top predictions (two shapes — backend may return either)
  top_3?: Array<{ label: string; probability: number }>;
  top_predictions?: Array<{ code: string; name: string; confidence: number }>;
  // Explainability — named dict (primary) or flat list (legacy)
  gradcam_available?: boolean;
  gradcam_images?: { original: string; gradcam: string; eigencam: string } | null;
  gradcam_images_list?: string[];  // legacy flat [orig, gradcam, eigencam]
  xai_error?: string | null;
  // U-Net Lesion Boundary Segmentation fields
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
  counterfactuals?: CounterfactualsMap | null;
  cf_error?: string | null;
  // Multimodal Patient Metadata Fusion fields
  metadata_fusion_available?: boolean;
  fusion_predicted_code?: string | null;
  fusion_predicted_name?: string | null;
  fusion_confidence?: number | null;
  fusion_agrees_with_image_only?: boolean | null;
  fusion_disagreement_note?: string | null;
  fusion_error?: string | null;
  image_quality_warning?: string | null;
  // Clinical Alert System fields
  alert_level?: "high_risk" | "low_confidence" | "normal";
  alert_message?: string | null;
  skin_tone_reliability_note?: string | null;
  ita_group?: string | null;
  ita_value?: number | null;
  // Offline / error
  error?: string;
  missing_path?: string;
  // Disclaimer
  disclaimer?: string;
}

export interface AssistantResult {
  ok?: boolean;
  answer: string;
  sources: Array<{ source: string; page: number | string }>;
  language_detected: string;
  language?: string;
  disclaimer?: string;
  // Citation-grounding & hallucination-verification fields
  answer_html?: string;
  citations?: Array<{
    marker: number;
    source: string;
    page: number | string;
    chunk_text_snippet: string;
  }>;
  sentences?: Array<{
    text: string;
    status: "SUPPORTED" | "PARTIAL" | "UNSUPPORTED";
    citation_markers?: number[];
  }>;
  verification_summary?: {
    total: number;
    supported: number;
    partial: number;
    unsupported: number;
  };
}

export interface PatientMetadataInput {
  patientAge?: number | null;
  patientSex?: string | null;
  patientLocalization?: string | null;
}

// ─── API calls ────────────────────────────────────────────────────────────────

/**
 * Check backend health and model/vectorstore availability.
 * Returns null if the backend is unreachable.
 */
export async function checkHealth(): Promise<HealthStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return null;
    return (await res.json()) as HealthStatus;
  } catch {
    return null;
  }
}

/**
 * Send an image file and optional patient metadata to the /predict endpoint.
 *
 * Returns a PredictResult dict always. The `available` boolean
 * indicates whether the model was loaded and the inference ran.
 * Returns { available: false } only on network failure.
 * Never returns a fake/demo result.
 */
export async function predictImage(
  file: File,
  includeCounterfactuals: boolean = false,
  metadata?: PatientMetadataInput
): Promise<PredictResult | null> {
  try {
    const form = new FormData();
    form.append("file", file);

    if (metadata?.patientAge !== undefined && metadata?.patientAge !== null && !isNaN(metadata.patientAge)) {
      form.append("patient_age", metadata.patientAge.toString());
    }
    if (metadata?.patientSex && metadata.patientSex !== "unknown") {
      form.append("patient_sex", metadata.patientSex);
    }
    if (metadata?.patientLocalization && metadata.patientLocalization !== "unknown") {
      form.append("patient_localization", metadata.patientLocalization);
    }

    const url = includeCounterfactuals
      ? `${API_BASE}/predict?include_counterfactuals=true`
      : `${API_BASE}/predict`;

    const res = await fetch(url, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(45000),
    });

    if (!res.ok) {
      // Non-2xx: try to parse the error body for a human message
      try {
        const errBody = await res.json();
        return {
          available: false,
          ok: false,
          error: errBody?.detail ?? `Server error (${res.status})`,
        };
      } catch {
        return { available: false, ok: false };
      }
    }

    const data = (await res.json()) as PredictResult;

    // Normalize `available`: the HF backend returns `ok` but may omit `available`.
    // Derive it: true if ok !== false AND at least one prediction field is present.
    const hasPrediction =
      Boolean(data.predicted_code) ||
      Boolean(data.predicted_name) ||
      Boolean(data.predicted_class) ||
      Boolean(data.predicted_label);

    return {
      ...data,
      available: data.available ?? (data.ok !== false && hasPrediction),
    };
  } catch {
    // Connection refused / backend not running
    return {
      available: false,
      ok: false,
      error: "Backend is not running. Start the FastAPI server on port 8000.",
    };
  }
}

/**
 * Ask the document-grounded assistant a question.
 *
 * Returns an AssistantResult on success.
 * Returns a clean offline AssistantResult when the backend or RAG is unavailable.
 * Never throws.
 */
export async function askAssistant(
  question: string,
  language: string = "auto"
): Promise<AssistantResult | null> {
  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, language }),
      signal: AbortSignal.timeout(60000),
    });

    if (!res.ok) {
      // Surface a clean offline result rather than returning null
      return {
        ok: false,
        answer: "Knowledge base is not connected in this environment.",
        sources: [],
        language_detected: "english",
      } as AssistantResult;
    }

    const data = (await res.json()) as AssistantResult;
    return data;
  } catch {
    return {
      ok: false,
      answer: "Knowledge base is not connected in this environment.",
      sources: [],
      language_detected: "english",
    } as AssistantResult;
  }
}
