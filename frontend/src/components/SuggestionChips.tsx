"use client";

import { GraduationCap, BookOpen, UserCheck, Building } from "lucide-react";

interface SuggestionChipsProps {
  onSelect: (prompt: string) => void;
  disabled?: boolean;
}

const SUGGESTIONS = [
  {
    icon: <GraduationCap className="w-3.5 h-3.5 text-indigo-400" />,
    label: "CSE R23 Syllabus",
    prompt: "What is the CSE R23 curriculum and syllabus structure?",
  },
  {
    icon: <UserCheck className="w-3.5 h-3.5 text-emerald-400" />,
    label: "HOD of CSE",
    prompt: "Who is the Head of the Department (HOD) of Computer Science and Engineering?",
  },
  {
    icon: <BookOpen className="w-3.5 h-3.5 text-amber-400" />,
    label: "Data Structures Credits",
    prompt: "What are the course outcomes and credits for Data Structures in R23?",
  },
  {
    icon: <Building className="w-3.5 h-3.5 text-cyan-400" />,
    label: "Central Library Facilities",
    prompt: "What facilities, books, and working hours does the SRKR Central Library have?",
  },
];

export function SuggestionChips({ onSelect, disabled }: SuggestionChipsProps) {
  return (
    <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-none no-scrollbar">
      {SUGGESTIONS.map((item, idx) => (
        <button
          key={idx}
          onClick={() => onSelect(item.prompt)}
          disabled={disabled}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-slate-900/80 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white transition-all whitespace-nowrap shadow-sm disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
        >
          {item.icon}
          <span>{item.label}</span>
        </button>
      ))}
    </div>
  );
}
