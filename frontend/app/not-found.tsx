import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-[#F5F7FB] flex items-center justify-center">
      <div className="text-center space-y-6 px-6">
        <div className="w-16 h-16 rounded-2xl bg-[#EFF6FF] border border-[#BFDBFE] flex items-center justify-center mx-auto">
          <span className="text-2xl font-black text-[#0B7FEA]">404</span>
        </div>
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-[#0F172A]">Page not found</h1>
          <p className="text-[#475569] text-sm max-w-xs">
            The page you are looking for does not exist or has been moved.
          </p>
        </div>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-6 py-2.5 bg-[#0B7FEA] text-white font-semibold text-sm rounded-xl hover:bg-[#0970d4] transition-colors"
        >
          Return to DermaLens AI
        </Link>
      </div>
    </div>
  );
}
