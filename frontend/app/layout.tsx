import type { Metadata, Viewport } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
  display: "swap",
});

export const metadata: Metadata = {
  title: "DermaLens AI — Clinical Skin Analysis Support",
  description:
    "AI-assisted skin lesion analysis built for safer clinical insight. Upload a clear skin image, review AI-assisted educational insights, and ask a document-grounded medical assistant follow-up questions.",
  keywords: [
    "skin lesion analysis",
    "AI dermatology",
    "melanoma detection",
    "clinical AI",
    "DermaLens",
  ],
  authors: [{ name: "DermaLens AI" }],
  robots: "index, follow",
  icons: {
    icon: "/title_photo.jpg",
    shortcut: "/title_photo.jpg",
    apple: "/title_photo.jpg",
  },
  openGraph: {
    title: "DermaLens AI — Clinical Skin Analysis Support",
    description:
      "AI-assisted skin lesion analysis built for safer clinical insight.",
    type: "website",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${outfit.variable}`}>
      <body className="font-sans bg-[#F8FAFC] text-[#0F172A] antialiased selection:bg-[#0B7FEA]/20 selection:text-[#0B7FEA]">
        {children}
      </body>
    </html>
  );
}
