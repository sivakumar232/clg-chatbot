"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatMessage } from "@/types/chat";
import { ReasoningSteps } from "./ReasoningSteps";
import { SourceBadge } from "./SourceBadge";
import { AlertCircle, Copy, Check, Sparkles, User } from "lucide-react";
import { useState } from "react";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`group flex items-start gap-2.5 py-2 px-1 sm:px-2 transition-colors ${
        isUser ? "flex-row-reverse" : "flex-row"
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-7 h-7 rounded-xl flex items-center justify-center shrink-0 text-xs font-semibold select-none shadow-xs ${
          isUser
            ? "bg-slate-800 text-white"
            : "bg-[#800020] text-white shadow-[#800020]/20"
        }`}
      >
        {isUser ? <User className="w-3.5 h-3.5" /> : <Sparkles className="w-3.5 h-3.5" />}
      </div>

      {/* Content */}
      <div
        className={`flex flex-col max-w-[85%] sm:max-w-[82%] ${
          isUser ? "items-end" : "items-start"
        }`}
      >
        {/* Meta Label */}
        <div className="flex items-center gap-1.5 mb-1 px-1">
          <span className="text-[11px] font-semibold text-slate-500">
            {isUser ? "You" : "SRKR Advisor"}
          </span>
          {!isUser && message.cacheHit && (
            <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 font-medium">
              ⚡ Cache
            </span>
          )}
        </div>

        {/* Bubble */}
        <div
          className={`relative rounded-2xl px-3.5 py-3 text-sm leading-relaxed transition-all ${
            isUser
              ? "bg-[#800020] text-white rounded-tr-xs shadow-xs"
              : "bg-white border border-slate-200/90 text-slate-800 rounded-tl-xs shadow-xs w-full"
          }`}
        >
          {/* Degraded mode warning */}
          {!isUser && message.degraded && (
            <div className="mb-2.5 p-2 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
              <span>
                <strong>Notice:</strong> Answering with reduced confidence based on nearest available records.
              </span>
            </div>
          )}

          {/* Reasoning Steps Accordion */}
          {!isUser && message.steps && message.steps.length > 0 && (
            <ReasoningSteps
              steps={message.steps}
              isStreaming={message.isStreaming}
            />
          )}

          {/* Markdown Content */}
          {message.content ? (
            <div
              className={`prose prose-sm max-w-none break-words leading-relaxed space-y-2 ${
                isUser ? "text-white prose-invert" : "text-slate-800"
              }`}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  table: ({ ...props }) => (
                    <div className="my-2.5 overflow-x-auto rounded-lg border border-slate-200 bg-white">
                      <table
                        className="w-full text-left text-xs border-collapse divide-y divide-slate-200"
                        {...props}
                      />
                    </div>
                  ),
                  thead: ({ ...props }) => (
                    <thead className="bg-slate-50 text-slate-700 font-semibold" {...props} />
                  ),
                  th: ({ ...props }) => (
                    <th className="px-3 py-2 border-b border-slate-200 font-semibold text-slate-800" {...props} />
                  ),
                  td: ({ ...props }) => (
                    <td className="px-3 py-2 border-b border-slate-100 text-slate-700" {...props} />
                  ),
                  code: ({ inline, className, children, ...props }: any) => {
                    if (inline) {
                      return (
                        <code
                          className={`px-1.5 py-0.5 rounded font-mono text-xs ${
                            isUser
                              ? "bg-rose-950/60 text-white"
                              : "bg-slate-100 text-[#800020] border border-slate-200 font-semibold"
                          }`}
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    }
                    return (
                      <div className="my-2 rounded-xl bg-slate-900 border border-slate-800 p-3 overflow-x-auto font-mono text-xs">
                        <code className="text-slate-200 block" {...props}>
                          {children}
                        </code>
                      </div>
                    );
                  },
                  ul: ({ ...props }) => (
                    <ul className="list-disc list-outside ml-4 space-y-1 my-1.5" {...props} />
                  ),
                  ol: ({ ...props }) => (
                    <ol className="list-decimal list-outside ml-4 space-y-1 my-1.5" {...props} />
                  ),
                  a: ({ ...props }) => (
                    <a
                      className={`${
                        isUser
                          ? "text-rose-200 underline font-medium"
                          : "text-[#800020] hover:underline font-semibold"
                      }`}
                      target="_blank"
                      rel="noopener noreferrer"
                      {...props}
                    />
                  ),
                  p: ({ ...props }) => <p className="mb-1.5 last:mb-0 leading-relaxed" {...props} />,
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          ) : null}

          {/* Sources Section */}
          {!isUser && message.sources && message.sources.length > 0 && (
            <SourceBadge sources={message.sources} />
          )}

          {/* Quick Copy Button */}
          {!isUser && message.content && (
            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={handleCopy}
                title="Copy response"
                className="p-1 rounded-md bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-500 hover:text-slate-800 transition-colors cursor-pointer"
              >
                {copied ? (
                  <Check className="w-3 h-3 text-emerald-600" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
