import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "App — DermaLens AI",
  description: "AI-assisted skin lesion analysis workspace.",
};

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden bg-[#F5F7FB]">
      {children}
    </div>
  );
}
