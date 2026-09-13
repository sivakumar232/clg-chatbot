"use client";

import { useState, useRef, useEffect } from "react";
import {
  MessageSquare,
  X,
  Sparkles,
  RotateCcw,
  Maximize2,
  Minimize2,
  Bot,
  ChevronDown,
} from "lucide-react";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
import { SuggestionChips } from "./SuggestionChips";
import { ChatMessage } from "@/types/chat";

interface ChatWidgetProps {
  messages: ChatMessage[];
  isLoading: boolean;
  isOpen: boolean;
  onToggle: () => void;
  onSend: (message: string) => void;
  onStop: () => void;
  onClear: () => void;
}

export function ChatWidget({
  messages,
  isLoading,
  isOpen,
  onToggle,
  onSend,
  onStop,
  onClear,
}: ChatWidgetProps) {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const [showTooltip, setShowTooltip] = useState<boolean>(true);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Auto scroll down when messages change and widget is open
  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isOpen]);

  return (
    <>
      {/* Floating Chatbot Window */}
      {isOpen && (
        <div
          className={`fixed z-50 bg-slate-50 border border-slate-200/90 shadow-2xl rounded-3xl flex flex-col overflow-hidden transition-all duration-300 animate-in fade-in slide-in-from-bottom-6 ${
            isExpanded
              ? "inset-4 sm:inset-10 w-auto h-auto max-w-5xl mx-auto"
              : "bottom-22 right-4 sm:right-6 w-[420px] max-w-[calc(100vw-2rem)] h-[620px] max-h-[calc(100vh-7rem)]"
          }`}
        >
          {/* Header in Burgundy #800020 */}
          <div className="bg-[#800020] text-white px-4 py-3.5 flex items-center justify-between shadow-xs select-none">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-white/15 border border-white/20 flex items-center justify-center text-white">
                <Sparkles className="w-4 h-4" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-bold tracking-tight">
                    Campus AI Assistant
                  </h3>
                  <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-white/20 font-mono">
                    RAG v2.0
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-[11px] text-rose-100">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span>College Knowledge Agent Online</span>
                </div>
              </div>
            </div>

            {/* Header Controls */}
            <div className="flex items-center gap-1">
              <button
                onClick={onClear}
                title="Reset Conversation"
                className="p-1.5 rounded-lg text-rose-200 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                title={isExpanded ? "Standard view" : "Expand view"}
                className="hidden sm:block p-1.5 rounded-lg text-rose-200 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
              >
                {isExpanded ? (
                  <Minimize2 className="w-3.5 h-3.5" />
                ) : (
                  <Maximize2 className="w-3.5 h-3.5" />
                )}
              </button>
              <button
                onClick={onToggle}
                title="Close chat"
                className="p-1.5 rounded-lg text-rose-200 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
              >
                <ChevronDown className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Conversation Feed */}
          <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-2.5 bg-slate-50">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Bottom Controls Area */}
          <div className="p-3 bg-white border-t border-slate-200/80 space-y-2">
            {/* Quick Suggestion Chips */}
            <SuggestionChips onSelect={onSend} disabled={isLoading} />

            {/* Input Component */}
            <ChatInput
              onSend={onSend}
              onStop={onStop}
              isLoading={isLoading}
              placeholder="Ask about syllabus, faculty, regulations..."
            />

            {/* Footer Note */}
            <div className="text-center text-[10px] text-slate-400 font-mono select-none">
              College Academic Assistant • Autonomous Knowledge Engine
            </div>
          </div>
        </div>
      )}

      {/* Floating Bottom-Right Launcher Button */}
      <div className="fixed bottom-6 right-6 z-50 flex items-center gap-3">
        {/* Helper speech bubble tooltip */}
        {!isOpen && showTooltip && (
          <div className="hidden sm:flex items-center gap-2 px-3.5 py-2 bg-white text-slate-800 text-xs font-medium rounded-2xl shadow-xl border border-slate-200/90 animate-bounce">
            <Sparkles className="w-3.5 h-3.5 text-[#800020]" />
            <span>Ask Campus AI</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowTooltip(false);
              }}
              className="text-slate-400 hover:text-slate-700 ml-1 cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        )}

        {/* The Bot Button */}
        <button
          onClick={onToggle}
          title={isOpen ? "Close AI Assistant" : "Open AI Assistant"}
          aria-label="Toggle Campus AI Assistant"
          className="relative w-14 h-14 rounded-full bg-[#800020] hover:bg-[#6b001b] text-white flex items-center justify-center shadow-xl hover:shadow-2xl active:scale-95 transition-all duration-200 cursor-pointer group"
        >
          {isOpen ? (
            <X className="w-6 h-6 transition-transform duration-200 group-hover:rotate-90" />
          ) : (
            <>
              <MessageSquare className="w-6 h-6 transition-transform duration-200 group-hover:scale-110" />
              {/* Online indicator ping */}
              <span className="absolute top-0.5 right-0.5 w-3.5 h-3.5 rounded-full bg-emerald-500 border-2 border-white shadow-xs" />
            </>
          )}
        </button>
      </div>
    </>
  );
}
