"use client";

import { FileText, ExternalLink } from "lucide-react";

interface SourceBadgeProps {
  sources: string[];
}

export function SourceBadge({ sources }: SourceBadgeProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 pt-3 border-t border-slate-800/80">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
        <FileText className="w-3.5 h-3.5 text-indigo-400" />
        <span>Verified Sources ({sources.length})</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {sources.map((src, idx) => {
          // If source is a URL, render clickable link, otherwise document badge
          const isUrl = src.startsWith("http://") || src.startsWith("https://");
          return (
            <div
              key={idx}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-lg bg-slate-800/60 border border-slate-700/60 text-slate-300 hover:bg-slate-800 hover:border-slate-600 transition-colors"
            >
              <FileText className="w-3 h-3 text-slate-400 shrink-0" />
              {isUrl ? (
                <a
                  href={src}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-indigo-300 underline underline-offset-2 flex items-center gap-1 truncate max-w-xs"
                >
                  <span className="truncate">{src}</span>
                  <ExternalLink className="w-2.5 h-2.5 shrink-0" />
                </a>
              ) : (
                <span className="truncate max-w-xs font-mono text-[11px]">{src}</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
