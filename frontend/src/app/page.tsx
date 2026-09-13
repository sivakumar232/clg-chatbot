"use client";

import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { ChatWidget } from "@/components/ChatWidget";
import { useAgentChat } from "@/hooks/useAgentChat";

export default function Home() {
  const { messages, isLoading, sendMessage, stopGeneration, clearChat } =
    useAgentChat();

  const [isChatOpen, setIsChatOpen] = useState<boolean>(true);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 selection:bg-[#800020]/15 selection:text-[#800020]">
      {/* Top Navbar */}
      <Navbar
        onOpenChat={() => setIsChatOpen(!isChatOpen)}
        isChatOpen={isChatOpen}
      />

      {/* Clean minimal backdrop without mock website contents */}
      <main className="flex-1 flex items-center justify-center p-4">
        {/* Ambient subtle center watermark when chat is minimized */}
        {!isChatOpen && (
          <div className="text-center space-y-2 select-none animate-in fade-in duration-300">
            <div className="w-12 h-12 rounded-2xl bg-[#800020]/10 text-[#800020] font-bold text-lg flex items-center justify-center mx-auto mb-3">
              AI
            </div>
            <h2 className="text-lg font-bold text-slate-800">
              Campus AI Academic Assistant
            </h2>
            <p className="text-xs text-slate-500 max-w-sm">
              Click the assistant button at the bottom-right corner to open the chat interface.
            </p>
          </div>
        )}
      </main>

      {/* Floating Bottom-Right Chatbot Widget */}
      <ChatWidget
        messages={messages}
        isLoading={isLoading}
        isOpen={isChatOpen}
        onToggle={() => setIsChatOpen(!isChatOpen)}
        onSend={(text) => sendMessage(text)}
        onStop={stopGeneration}
        onClear={clearChat}
      />
    </div>
  );
}
