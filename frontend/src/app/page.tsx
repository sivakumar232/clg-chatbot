"use client";

import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { ChatWidget } from "@/components/ChatWidget";
import { useAgentChat } from "@/hooks/useAgentChat";
import {
  GraduationCap,
  Users,
  BookOpen,
  Building2,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Award,
  Search,
} from "lucide-react";

export default function Home() {
  const { messages, isLoading, sendMessage, stopGeneration, clearChat } =
    useAgentChat();

  const [isChatOpen, setIsChatOpen] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>("");

  const handleQuickQuestion = (prompt: string) => {
    setIsChatOpen(true);
    sendMessage(prompt);
  };

  const handleHeroSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setIsChatOpen(true);
    sendMessage(searchQuery);
    setSearchQuery("");
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#fbfcfd] text-slate-900 selection:bg-[#800020]/15 selection:text-[#800020]">
      {/* Top Collegiate Navbar */}
      <Navbar
        onOpenChat={() => setIsChatOpen(!isChatOpen)}
        isChatOpen={isChatOpen}
      />

      {/* Main Campus Portal Body */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-8 sm:py-12 space-y-12">
        {/* Hero Section */}
        <div className="text-center space-y-4 max-w-3xl mx-auto pt-2 sm:pt-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#800020]/10 border border-[#800020]/20 text-[#800020] text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Official College AI Assistant • LangGraph & Qdrant RAG</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Explore SRKR Engineering College <br className="hidden sm:inline" />
            <span className="text-[#800020]">Curriculum & Campus Knowledge</span>
          </h1>

          <p className="text-sm sm:text-base text-slate-600 max-w-2xl mx-auto leading-relaxed">
            Instant, verified information extracted directly from autonomous R23/R20
            curriculum documents, department faculty directories, and college regulations.
          </p>

          {/* Quick Search Bar triggering the assistant */}
          <form
            onSubmit={handleHeroSearch}
            className="mt-6 max-w-xl mx-auto relative flex items-center shadow-md rounded-2xl overflow-hidden border border-slate-300 focus-within:border-[#800020] focus-within:ring-2 focus-within:ring-[#800020]/15 bg-white transition-all"
          >
            <Search className="w-5 h-5 text-slate-400 ml-4 shrink-0" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Ask anything (e.g. CSE R23 syllabus, HOD contacts, credit rules)..."
              className="w-full px-3 py-3.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none bg-transparent"
            />
            <button
              type="submit"
              className="m-1.5 px-4 py-2 rounded-xl bg-[#800020] hover:bg-[#6b001b] text-white text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer shrink-0"
            >
              <span>Ask AI</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </form>
        </div>

        {/* 4 Interactive Starter Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            {
              title: "CSE R23 Curriculum",
              desc: "Full course structure, core subjects, electives, and semester credit weights.",
              prompt: "What is the CSE R23 curriculum and course credit structure?",
              icon: GraduationCap,
            },
            {
              title: "Faculty Directory",
              desc: "Department heads, professors, qualifications, research domains, and contacts.",
              prompt:
                "Who is the Head of Department (HOD) and key faculty of Computer Science?",
              icon: Users,
            },
            {
              title: "Course Syllabi & Credits",
              desc: "Prerequisites, syllabus units, and examination outcomes for core courses.",
              prompt:
                "What are the course outcomes and syllabus modules for Data Structures in R23?",
              icon: BookOpen,
            },
            {
              title: "Campus & Labs",
              desc: "Library facilities, digital subscriptions, computing center, and college hours.",
              prompt:
                "What facilities, books, and working hours does the SRKR Central Library have?",
              icon: Building2,
            },
          ].map((item, idx) => {
            const Icon = item.icon;
            return (
              <button
                key={idx}
                onClick={() => handleQuickQuestion(item.prompt)}
                className="group p-5 rounded-2xl bg-white border border-slate-200/90 hover:border-[#800020]/50 hover:shadow-md transition-all text-left flex flex-col justify-between cursor-pointer"
              >
                <div>
                  <div className="w-9 h-9 rounded-xl bg-[#800020]/10 text-[#800020] flex items-center justify-center mb-3 group-hover:bg-[#800020] group-hover:text-white transition-colors">
                    <Icon className="w-5 h-5" />
                  </div>
                  <h3 className="text-sm font-bold text-slate-800 group-hover:text-[#800020] transition-colors">
                    {item.title}
                  </h3>
                  <p className="mt-1.5 text-xs text-slate-500 leading-relaxed">
                    {item.desc}
                  </p>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs font-semibold text-[#800020]">
                  <span>Ask Assistant</span>
                  <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
                </div>
              </button>
            );
          })}
        </div>

        {/* Academic Accreditation & Highlights Banner */}
        <div className="rounded-2xl bg-white border border-slate-200/90 p-6 sm:p-8 flex flex-col md:flex-row items-center justify-between gap-6 shadow-xs">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl bg-[#800020] text-white flex items-center justify-center shrink-0">
              <Award className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Sagi Rama Krishnam Raju Engineering College (SRKREC)
              </h3>
              <p className="text-xs text-slate-600 mt-0.5">
                Autonomous Institution • Affiliated to JNTUK Kakinada • Approved by AICTE, New Delhi
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <span className="px-3 py-1 rounded-lg bg-slate-100 text-slate-700 text-xs font-semibold border border-slate-200">
              NAAC A+ Grade
            </span>
            <span className="px-3 py-1 rounded-lg bg-slate-100 text-slate-700 text-xs font-semibold border border-slate-200">
              NBA Accredited
            </span>
            <span className="px-3 py-1 rounded-lg bg-[#800020]/10 text-[#800020] text-xs font-semibold border border-[#800020]/20">
              Estd. 1980
            </span>
          </div>
        </div>
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
