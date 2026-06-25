import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-[#E5EAF0] bg-white">
      <div className="max-w-7xl mx-auto px-6 py-10">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo + tagline */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-[10px] bg-[#0F172A] flex items-center justify-center">
              <span className="text-white text-[11px] font-bold tracking-wider">DL</span>
            </div>
            <div>
              <span className="font-display text-[15px] font-semibold text-[#0F172A] tracking-tight">
                DermaLens <span className="text-[#64748B] font-medium">AI</span>
              </span>
              <p className="text-[11px] text-[#94A3B8] tracking-wide mt-0.5">Clinical skin analysis support</p>
            </div>
          </div>

          {/* Links */}
          <div className="flex items-center gap-6">
            <Link href="#how-it-works" className="text-xs text-[#475569] hover:text-[#0F172A] transition-colors">
              How it works
            </Link>
            <Link href="#safety" className="text-xs text-[#475569] hover:text-[#0F172A] transition-colors">
              Safety
            </Link>
            <Link href="/app" className="text-xs text-[#475569] hover:text-[#0F172A] transition-colors">
              App
            </Link>
          </div>

          {/* Copyright */}
          <p className="text-xs text-[#94A3B8] text-center">
            &copy; {new Date().getFullYear()} DermaLens AI. Educational use only. Not a clinical diagnostic tool.
          </p>
        </div>
      </div>
    </footer>
  );
}
