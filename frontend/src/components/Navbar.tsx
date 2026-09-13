"use client";

import { useEffect, useState } from "react";
import { MessageSquare, ShieldCheck, Sparkles } from "lucide-react";

interface NavbarProps {
  onOpenChat?: () => void;
  isChatOpen?: boolean;
}

export function Navbar({ onOpenChat, isChatOpen }: NavbarProps) {
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${apiUrl}/api/health`, { method: "GET" });
        setIsBackendHealthy(res.ok);
      } catch {
        setIsBackendHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-30 w-full border-b border-slate-200/80 bg-white/95 backdrop-blur-md px-4 py-3 sm:px-6 shadow-xs">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Left Branding */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#800020] text-white flex items-center justify-center font-bold text-sm tracking-wider shadow-sm select-none">
            SRKR
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-slate-900 tracking-tight leading-none">
                S.R.K.R. Engineering College
              </h1>
              <span className="hidden sm:inline-flex text-[10px] px-2 py-0.5 rounded-full bg-[#800020]/10 text-[#800020] font-semibold border border-[#800020]/20">
                Autonomous
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium mt-0.5">
              Affiliated to JNTUK • Accredited NAAC A+ • Bhimavaram
            </p>
          </div>
        </div>

        {/* Right Actions & Status */}
        <div className="flex items-center gap-3">
          {/* Health Status Indicator */}
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs text-slate-600 font-medium">
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendHealthy === null
                  ? "bg-slate-400 animate-pulse"
                  : isBackendHealthy
                  ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"
                  : "bg-rose-500"
              }`}
            />
            <span className="text-[11px]">
              {isBackendHealthy === null
                ? "Connecting..."
                : isBackendHealthy
                ? "RAG Agent Online"
                : "Agent Offline"}
            </span>
          </div>

          {/* Quick Chat Open CTA */}
          {onOpenChat && (
            <button
              onClick={onOpenChat}
              className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-[#800020] hover:bg-[#6b001b] text-white text-xs font-semibold shadow-sm hover:shadow transition-all active:scale-95"
            >
              <Sparkles className="w-3.5 h-3.5 text-rose-200" />
              <span>{isChatOpen ? "Bot Open" : "Ask AI Bot"}</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
