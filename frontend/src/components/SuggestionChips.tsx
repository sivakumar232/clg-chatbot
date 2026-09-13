"use client";

import { GraduationCap, BookOpen, UserCheck, Building2 } from "lucide-react";

interface SuggestionChipsProps {
  onSelect: (prompt: string) => void;
  disabled?: boolean;
}

const SUGGESTIONS = [
  {
    icon: GraduationCap,
    label: "CSE R23 Syllabus",
    prompt: "What is the CSE R23 curriculum and syllabus structure?",
  },
  {
    icon: UserCheck,
    label: "CSE HOD & Staff",
    prompt: "Who is the Head of Department (HOD) of Computer Science and Engineering?",
  },
  {
    icon: BookOpen,
    label: "Data Structures Credits",
    prompt: "What are the course outcomes and credits for Data Structures in R23?",
  },
  {
    icon: Building2,
    label: "Library & Labs",
    prompt: "What facilities and timings does the Central Library have?",
  },
];

export function SuggestionChips({ onSelect, disabled }: SuggestionChipsProps) {
  return (
    <div className="flex items-center gap-1.5 overflow-x-auto pb-1.5 scrollbar-none no-scrollbar">
      {SUGGESTIONS.map((item, idx) => {
        const Icon = item.icon;
        return (
          <button
            key={idx}
            onClick={() => onSelect(item.prompt)}
            disabled={disabled}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-white hover:bg-[#800020]/5 border border-slate-200 hover:border-[#800020]/40 text-slate-700 hover:text-[#800020] transition-all whitespace-nowrap shadow-xs disabled:opacity-40 disabled:cursor-not-allowed shrink-0 cursor-pointer"
          >
            <Icon className="w-3.5 h-3.5 text-[#800020] shrink-0" />
            <span>{item.label}</span>
          </button>
        );
      })}
    </div>
  );
}
