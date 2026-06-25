"use client";

import Link from "next/link";
import clsx from "clsx";
import {
  ScanLine,
  Activity,
  Eye,
  MessageSquare,
  BookOpen,
  ArrowLeft,
} from "lucide-react";

const navItems = [
  { label: "Analyze", href: "/app", icon: ScanLine, id: "sidebar-analyze" },
  { label: "Result", href: "/app?panel=result", icon: Activity, id: "sidebar-result" },
  { label: "Explain", href: "/app?panel=explain", icon: Eye, id: "sidebar-explain" },
  { label: "Ask Assistant", href: "/app?panel=assistant", icon: MessageSquare, id: "sidebar-assistant" },
  { label: "Guide", href: "/app?panel=guide", icon: BookOpen, id: "sidebar-guide" },
];

interface SidebarProps {
  activePanel: string;
  onPanelChange: (panel: string) => void;
}

export default function Sidebar({ activePanel, onPanelChange }: SidebarProps) {
  return (
    <aside className="w-64 min-h-screen bg-[#F8FAFC] border-r border-[#E2E8F0] flex flex-col flex-shrink-0">
      {/* Logo */}
      <div className="h-16 flex items-center px-6 border-b border-[#E2E8F0]">
        <div className="flex items-center gap-3 group">
          <div className="w-8 h-8 rounded-[8px] bg-[#0F172A] flex items-center justify-center shadow-sm">
            <span className="text-white text-[11px] font-bold tracking-wider">DL</span>
          </div>
          <div>
            <p className="font-display text-[15px] font-semibold text-[#0F172A] leading-none tracking-tight">
              DermaLens <span className="text-[#64748B] font-medium">AI</span>
            </p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-4 py-6 space-y-1">
        <p className="px-3 mb-3 text-[10px] font-semibold text-[#94A3B8] uppercase tracking-[0.15em]">
          Workspace
        </p>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePanel === item.label.toLowerCase().replace(" ", "-");
          return (
            <button
              key={item.label}
              id={item.id}
              onClick={() => onPanelChange(item.label.toLowerCase().replace(" ", "-"))}
              className={clsx(
                "relative w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all duration-200 text-left group",
                isActive
                  ? "bg-[#F1F5F9] text-[#0B7FEA]"
                  : "text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#0F172A]"
              )}
            >
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 bg-[#0B7FEA] rounded-r-full" />
              )}
              <Icon
                size={16}
                strokeWidth={isActive ? 2 : 1.5}
                className={clsx(
                  "transition-colors duration-200",
                  isActive ? "text-[#0B7FEA]" : "text-[#94A3B8] group-hover:text-[#64748B]"
                )}
              />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Bottom — back to home */}
      <div className="p-4">
        <Link
          href="/"
          id="sidebar-home"
          className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[13px] text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#0F172A] transition-colors"
        >
          <ArrowLeft size={16} className="text-[#94A3B8]" />
          Exit to home
        </Link>
      </div>
    </aside>
  );
}
