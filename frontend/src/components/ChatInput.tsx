"use client";

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop: () => void;
  isLoading: boolean;
}

export function ChatInput({ onSend, onStop, isLoading }: ChatInputProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto resize textarea
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
    <div className="relative flex items-end gap-2 bg-slate-900/90 border border-slate-800 rounded-2xl p-2 shadow-2xl backdrop-blur-md focus-within:border-indigo-500/80 transition-colors">
      <textarea
        ref={textareaRef}
        rows={1}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything about SRKR Engineering College (e.g. syllabus, faculty, regulations)..."
        className="w-full resize-none bg-transparent px-3 py-1.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none min-h-[40px] max-h-[140px] leading-relaxed"
      />

      <div className="shrink-0 flex items-center mb-0.5">
        {isLoading ? (
          <button
            type="button"
            onClick={onStop}
            title="Stop generation"
            className="p-2.5 rounded-xl bg-rose-600/90 hover:bg-rose-600 text-white transition-all shadow-md flex items-center justify-center"
          >
            <Square className="w-4 h-4 fill-white" />
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSend}
            disabled={!text.trim()}
            title="Send query"
            className="p-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 text-white transition-all shadow-md flex items-center justify-center"
          >
            <Send className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}
