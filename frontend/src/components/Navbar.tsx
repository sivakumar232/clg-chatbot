"use client";

import { useEffect, useState } from "react";
import { Sparkles, Trash2, Cpu, ExternalLink } from "lucide-react";

interface NavbarProps {
  onClear: () => void;
}

export function Navbar({ onClear }: NavbarProps) {
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${apiUrl}/api/health`);
        if (res.ok) {
          setIsBackendHealthy(true);
        } else {
          setIsBackendHealthy(false);
        }
      } catch {
        setIsBackendHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-30 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md px-4 py-3">
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        {/* Left Branding */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-purple-500 flex items-center justify-center text-white shadow-md shadow-indigo-600/20">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm md:text-base font-bold text-white tracking-tight">
                SRKR Agentic Assistant
              </h1>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-800 font-medium">
                RAG v2.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              S.R.K.R. Engineering College (Autonomous) • Bhimavaram
            </p>
          </div>
        </div>

        {/* Right Info & Actions */}
        <div className="flex items-center gap-3">
          {/* Health Status Indicator */}
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-300">
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendHealthy === null
                  ? "bg-slate-500 animate-ping"
                  : isBackendHealthy
                  ? "bg-emerald-400 animate-pulse"
                  : "bg-rose-500"
              }`}
            />
            <span className="text-[11px] font-mono">
              {isBackendHealthy === null
                ? "Checking..."
                : isBackendHealthy
                ? "API Connected"
                : "API Offline"}
            </span>
          </div>

          {/* Model Tag */}
          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 text-xs font-mono">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>Qwen 2.5-32B / Jina v2</span>
          </div>

          {/* Clear Button */}
          <button
            onClick={onClear}
            title="Reset Chat"
            className="p-2 rounded-xl text-slate-400 hover:text-rose-400 hover:bg-slate-900 transition-colors border border-transparent hover:border-slate-800"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
