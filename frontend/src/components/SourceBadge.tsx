"use client";

import { FileText, ExternalLink } from "lucide-react";

interface SourceBadgeProps {
  sources: string[];
}

export function SourceBadge({ sources }: SourceBadgeProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3.5 pt-3 border-t border-slate-200">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
        <FileText className="w-3.5 h-3.5 text-[#800020]" />
        <span>Verified Sources ({sources.length})</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((src, idx) => {
          const isUrl = src.startsWith("http://") || src.startsWith("https://");
          return (
            <div
              key={idx}
              className="inline-flex items-center gap-1 px-2.5 py-1 text-xs rounded-lg bg-slate-50 border border-slate-200 text-slate-700 hover:border-[#800020]/40 hover:bg-[#800020]/5 hover:text-[#800020] transition-colors"
            >
              <FileText className="w-3 h-3 text-slate-400 shrink-0" />
              {isUrl ? (
                <a
                  href={src}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:underline flex items-center gap-1 truncate max-w-xs text-[11px] font-medium"
                >
                  <span className="truncate">{src}</span>
                  <ExternalLink className="w-2.5 h-2.5 shrink-0 text-slate-400" />
                </a>
              ) : (
                <span className="truncate max-w-xs text-[11px] font-mono">
                  {src}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
