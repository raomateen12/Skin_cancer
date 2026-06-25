"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { Menu, X } from "lucide-react";

const navLinks = [
  { label: "How it works", href: "#how-it-works" },
  { label: "Safety", href: "#safety" },
  { label: "Assistant", href: "/app" },
  { label: "Guide", href: "/app" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/90 backdrop-blur-md border-b border-[#E5EAF0] shadow-sm"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-[10px] bg-[#0F172A] flex items-center justify-center shadow-sm">
            <span className="text-white text-xs font-bold tracking-wider">DL</span>
          </div>
          <span className="font-display text-[#0F172A] font-semibold text-[17px] tracking-tight">
            DermaLens <span className="text-[#64748B] font-medium">AI</span>
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-2">
          {navLinks.map((link) => (
            <Link
              key={link.label}
              href={link.href}
              className="px-4 py-2 text-[14px] font-medium text-[#475569] hover:text-[#0F172A] hover:bg-[#F8FAFC] rounded-xl transition-all duration-300"
            >
              {link.label}
            </Link>
          ))}
        </nav>

        {/* CTA */}
        <div className="hidden md:flex items-center gap-3">
          <Link
            href="/app"
            id="navbar-get-started"
            className="px-6 py-2.5 bg-[#0F172A] hover:bg-[#1E293B] text-white text-[14px] font-medium rounded-xl shadow-[0_2px_10px_rgba(15,23,42,0.1)] transition-all duration-300 hover:shadow-[0_4px_14px_rgba(15,23,42,0.15)]"
          >
            Get Started
          </Link>
        </div>

        {/* Mobile hamburger */}
        <button
          className="md:hidden p-2 rounded-lg text-[#475569] hover:bg-[#F5F7FB] transition-colors"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation"
        >
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="md:hidden bg-white border-t border-[#E5EAF0] px-6 py-4 flex flex-col gap-1 animate-fade-in">
          {navLinks.map((link) => (
            <Link
              key={link.label}
              href={link.href}
              onClick={() => setMobileOpen(false)}
              className="px-4 py-3 text-sm font-medium text-[#475569] hover:text-[#0F172A] hover:bg-[#F5F7FB] rounded-lg transition-colors"
            >
              {link.label}
            </Link>
          ))}
          <Link
            href="/app"
            onClick={() => setMobileOpen(false)}
            className="mt-2 px-5 py-3 bg-[#0B7FEA] text-white text-sm font-semibold rounded-lg text-center"
          >
            Get Started
          </Link>
        </div>
      )}

      {/* Ticker */}
      <div className="w-full bg-[#0B7FEA]/10 border-b border-[#E5EAF0] flex items-center justify-center h-8">
        <div className="text-[12px] font-medium text-[#0B7FEA] tracking-wider uppercase text-center">
          Build with 💜 By <a href="https://www.linkedin.com/in/mateen-ahmad-saeed-8a0574252/?originalSubdomain=pk" target="_blank" rel="noopener noreferrer" className="underline hover:text-[#0970d4] transition-colors">Mateen</a>
        </div>
      </div>
    </header>
  );
}
