"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatMessage } from "@/types/chat";
import { ReasoningSteps } from "./ReasoningSteps";
import { SourceBadge } from "./SourceBadge";
import { Bot, User, AlertTriangle, Sparkles, Copy, Check } from "lucide-react";
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
      className={`group flex items-start gap-3.5 py-4 px-2 md:px-4 transition-colors ${
        isUser ? "flex-row-reverse" : "flex-row"
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 shadow-md ${
          isUser
            ? "bg-gradient-to-br from-indigo-500 to-indigo-700 text-white"
            : "bg-gradient-to-br from-indigo-600 via-indigo-700 to-purple-800 text-indigo-100 border border-indigo-400/30"
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Message Content Container */}
      <div
        className={`flex flex-col max-w-[88%] md:max-w-3xl ${
          isUser ? "items-end" : "items-start"
        }`}
      >
        {/* Author Label */}
        <div className="flex items-center gap-2 mb-1.5 px-1">
          <span className="text-xs font-semibold text-slate-400">
            {isUser ? "You" : "SRKR Advisor"}
          </span>
          {!isUser && message.cacheHit && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-700/60 font-medium">
              ⚡ Cache Hit
            </span>
          )}
          {!isUser && message.provider && (
            <span className="text-[10px] text-slate-500 font-mono">
              • {message.provider}
            </span>
          )}
        </div>

        {/* Bubble */}
        <div
          className={`relative rounded-2xl px-4 py-3.5 shadow-sm text-sm leading-relaxed ${
            isUser
              ? "bg-indigo-600 text-white rounded-tr-none font-normal"
              : "bg-slate-900/90 border border-slate-800/90 text-slate-100 rounded-tl-none backdrop-blur-sm shadow-xl"
          }`}
        >
          {/* Degraded mode disclaimer */}
          {!isUser && message.degraded && (
            <div className="mb-3 p-2.5 rounded-lg bg-amber-950/50 border border-amber-800/60 text-amber-200 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span>
                <strong>Notice:</strong> Answering with reduced confidence based on closest retrieved context.
              </span>
            </div>
          )}

          {/* Reasoning Steps (Active/Completed Accordion) */}
          {!isUser && message.steps && message.steps.length > 0 && (
            <ReasoningSteps
              steps={message.steps}
              isStreaming={message.isStreaming}
            />
          )}

          {/* Markdown Content */}
          {message.content ? (
            <div className="prose prose-invert prose-sm max-w-none break-words leading-relaxed space-y-2">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  table: ({ ...props }) => (
                    <div className="my-3 overflow-x-auto rounded-lg border border-slate-800">
                      <table
                        className="w-full text-left text-xs border-collapse divide-y divide-slate-800"
                        {...props}
                      />
                    </div>
                  ),
                  thead: ({ ...props }) => (
                    <thead className="bg-slate-950 text-slate-300 font-semibold" {...props} />
                  ),
                  th: ({ ...props }) => (
                    <th className="px-3 py-2 border-b border-slate-800" {...props} />
                  ),
                  td: ({ ...props }) => (
                    <td className="px-3 py-2 border-b border-slate-800/60 text-slate-300" {...props} />
                  ),
                  code: ({ inline, className, children, ...props }: any) => {
                    if (inline) {
                      return (
                        <code
                          className="px-1.5 py-0.5 rounded bg-slate-800 text-indigo-300 font-mono text-xs"
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    }
                    return (
                      <div className="my-2 rounded-xl bg-slate-950 border border-slate-800 p-3 overflow-x-auto">
                        <code className="text-xs font-mono text-emerald-300 block" {...props}>
                          {children}
                        </code>
                      </div>
                    );
                  },
                  ul: ({ ...props }) => (
                    <ul className="list-disc list-inside space-y-1 my-2 text-slate-200" {...props} />
                  ),
                  ol: ({ ...props }) => (
                    <ol className="list-decimal list-inside space-y-1 my-2 text-slate-200" {...props} />
                  ),
                  a: ({ ...props }) => (
                    <a
                      className="text-indigo-400 hover:text-indigo-300 underline underline-offset-2"
                      target="_blank"
                      rel="noopener noreferrer"
                      {...props}
                    />
                  ),
                  p: ({ ...props }) => <p className="mb-2 last:mb-0 leading-normal" {...props} />,
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
                className="p-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
