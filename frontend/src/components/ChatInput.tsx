"use client";

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { ArrowUp, Square } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop: () => void;
  isLoading: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  onStop,
  isLoading,
  placeholder = "Ask about SRKR syllabus, faculty, regulations, or credits...",
}: ChatInputProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto resize textarea smoothly
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        140
      )}px`;
    }
  }, [text]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (!text.trim() || isLoading) return;
    onSend(text);
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  return (
    <div className="relative rounded-2xl border border-slate-300 bg-white shadow-xs focus-within:border-[#800020] focus-within:ring-2 focus-within:ring-[#800020]/15 transition-all">
      <div className="flex flex-col p-2 sm:p-2.5">
        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="w-full resize-none bg-transparent px-2 py-1 text-xs sm:text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none min-h-[36px] max-h-[140px] leading-relaxed"
        />

        <div className="flex items-center justify-between mt-1 pt-1.5 border-t border-slate-100 px-1 text-xs">
          <div className="text-[10px] text-slate-400 font-mono hidden sm:inline">
            <kbd className="px-1 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-500">
              Enter
            </kbd>{" "}
            to send
          </div>
          <div className="sm:hidden" />

          <div className="flex items-center gap-1.5">
            {isLoading ? (
              <button
                type="button"
                onClick={onStop}
                title="Stop generation"
                className="flex items-center gap-1 px-2.5 py-1 rounded-xl bg-slate-800 hover:bg-slate-900 text-white text-xs font-medium transition-all active:scale-95 shadow-xs cursor-pointer"
              >
                <Square className="w-3 h-3 fill-rose-400 text-rose-400" />
                <span className="text-[10px]">Stop</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!text.trim()}
                title="Send query"
                className="flex items-center justify-center w-7 h-7 sm:w-8 sm:h-8 rounded-xl bg-[#800020] hover:bg-[#6b001b] disabled:bg-slate-200 disabled:text-slate-400 text-white transition-all active:scale-95 disabled:active:scale-100 shadow-xs cursor-pointer disabled:cursor-not-allowed"
              >
                <ArrowUp className="w-4 h-4 stroke-[2.5]" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
