"use client";

import { useEffect, useRef } from "react";
import { Navbar } from "@/components/Navbar";
import { MessageBubble } from "@/components/MessageBubble";
import { ChatInput } from "@/components/ChatInput";
import { SuggestionChips } from "@/components/SuggestionChips";
import { useAgentChat } from "@/hooks/useAgentChat";
import { GraduationCap, Sparkles, BookOpen, Layers, ShieldCheck } from "lucide-react";

export default function Home() {
  const { messages, isLoading, sendMessage, stopGeneration, clearChat } =
    useAgentChat();

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Auto scroll to bottom when new messages or reasoning steps arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-gradient-to-b from-[#090D16] via-[#0D1322] to-[#0A0E1A]">
      {/* Top Navigation */}
      <Navbar onClear={clearChat} />

      {/* Main Chat Container */}
      <main className="flex-1 overflow-y-auto px-2 md:px-4 py-4 space-y-2 max-w-4xl w-full mx-auto">
        {/* Messages List */}
        <div className="space-y-1">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Bottom Sticky Control Section */}
      <footer className="w-full bg-slate-950/80 backdrop-blur-xl border-t border-slate-800/80 px-4 py-3 shrink-0">
        <div className="max-w-4xl mx-auto space-y-2.5">
          {/* Quick suggestions */}
          <SuggestionChips
            onSelect={(prompt) => sendMessage(prompt)}
            disabled={isLoading}
          />

          {/* User Input & Send Button */}
          <ChatInput
            onSend={(text) => sendMessage(text)}
            onStop={stopGeneration}
            isLoading={isLoading}
          />

          {/* Footer note */}
          <div className="text-center text-[11px] text-slate-500 flex items-center justify-center gap-2">
            <span>SRKR Agentic RAG • LangGraph State Machine • Qdrant Vector + BM25 Fusion</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
