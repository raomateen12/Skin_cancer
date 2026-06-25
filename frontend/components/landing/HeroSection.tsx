import Link from "next/link";
import { ArrowRight, ChevronDown } from "lucide-react";
import ProductMockup from "./ProductMockup";

export default function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center overflow-hidden bg-[#F5F7FB]">
      {/* Subtle background grid */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `linear-gradient(#E5EAF0 1px, transparent 1px), linear-gradient(90deg, #E5EAF0 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
          opacity: 0.4,
        }}
      />

      {/* Top-right gradient blob */}
      <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-gradient-to-bl from-[#0B7FEA]/8 via-[#16B8C7]/6 to-transparent rounded-full blur-3xl pointer-events-none" />
      {/* Bottom-left gradient blob */}
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-gradient-to-tr from-[#16B8C7]/6 to-transparent rounded-full blur-3xl pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-6 pt-36 md:pt-40 pb-16 w-full">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left column */}
          <div className="space-y-8 animate-fade-up">
            {/* Headline */}
            <div className="space-y-4">
              <h1 className="text-4xl md:text-5xl xl:text-6xl font-bold text-[#0F172A] leading-[1.08] tracking-tight">
                AI-assisted skin{" "}
                <span className="text-gradient">lesion analysis,</span>{" "}
                built for safer clinical insight.
              </h1>
              <p className="text-lg text-[#475569] leading-relaxed max-w-[520px]">
                Upload a clear skin image, review AI-assisted educational insights, and ask
                a document-grounded medical assistant follow-up questions.
              </p>
            </div>

            {/* CTA buttons */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-4">
              <Link
                href="/app"
                id="hero-get-started"
                className="group inline-flex items-center justify-center gap-2 px-7 py-3.5 bg-[#0B7FEA] hover:bg-[#0970d4] text-white font-semibold rounded-xl shadow-md hover:shadow-lg transition-all duration-200"
              >
                Get Started
                <ArrowRight
                  size={16}
                  className="group-hover:translate-x-0.5 transition-transform duration-150"
                />
              </Link>
              <a
                href="#how-it-works"
                id="hero-learn-more"
                className="inline-flex items-center justify-center gap-2 px-7 py-3.5 bg-white hover:bg-[#F5F7FB] border border-[#E5EAF0] text-[#0F172A] font-semibold rounded-xl shadow-sm hover:shadow-md transition-all duration-200"
              >
                Learn how it works
                <ChevronDown size={16} className="text-[#475569]" />
              </a>
            </div>

            {/* Trust indicators */}
            <div className="flex flex-wrap items-center gap-6 pt-2">
              {[
                { label: "Educational use only" },
                { label: "Explainability layer" },
                { label: "Document-grounded" },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#0B7FEA]" fill="none" viewBox="0 0 16 16">
                    <path
                      d="M13.5 4.5L6.5 11.5L2.5 7.5"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span className="text-sm text-[#475569] font-medium">{item.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Right column — product mockup */}
          <div className="flex justify-center lg:justify-end animate-fade-up delay-200">
            <ProductMockup />
          </div>
        </div>
      </div>
    </section>
  );
}
