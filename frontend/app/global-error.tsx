"use client";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "system-ui, sans-serif",
          background: "#F5F7FB",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "100vh",
        }}
      >
        <div style={{ textAlign: "center", padding: "2rem" }}>
          <h1 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#0F172A", marginBottom: "0.75rem" }}>
            Something went wrong
          </h1>
          <p style={{ fontSize: "0.875rem", color: "#475569", marginBottom: "1.5rem" }}>
            {error.message ?? "An unexpected error occurred."}
          </p>
          <button
            onClick={reset}
            style={{
              padding: "0.625rem 1.5rem",
              background: "#0B7FEA",
              color: "white",
              fontWeight: 600,
              fontSize: "0.875rem",
              borderRadius: "0.75rem",
              border: "none",
              cursor: "pointer",
            }}
          >
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}
