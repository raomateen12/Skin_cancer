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

export interface PredictResult {
  ok?: boolean;
  available?: boolean;  // optional — backend may omit this; derived below if absent
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
  // Explainability
  gradcam_available?: boolean;
  gradcam_images?: string[];
  // Offline / error
  error?: string;
  missing_path?: string;
  // Disclaimer
  disclaimer?: string;
}

export interface AssistantResult {
  ok?: boolean;
  answer: string;
  sources: Array<{ source: string; page: number }>;
  language_detected: string;
  language?: string;
  disclaimer?: string;
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
 * Send an image file to the /predict endpoint.
 *
 * Returns a PredictResult dict always. The `available` boolean
 * indicates whether the model was loaded and the inference ran.
 * Returns { available: false } only on network failure.
 * Never returns a fake/demo result.
 */
export async function predictImage(file: File): Promise<PredictResult | null> {
  try {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(30000),
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
      signal: AbortSignal.timeout(20000),
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
