"use client";

import {
  GraduationCap,
  Users,
  Layers,
  Building2,
  ArrowUpRight,
  Sparkles,
} from "lucide-react";
import { ShimmerBadge } from "./magicui/ShimmerBadge";

interface HeroSuggestionsProps {
  onSelect: (prompt: string) => void;
  disabled?: boolean;
}

const STARTER_CARDS = [
  {
    category: "Curriculum R23",
    title: "Syllabus & Regulations",
    description:
      "Explore course structure, elective subjects, credit distribution & prerequisites.",
    prompt: "What is the CSE R23 curriculum and course credit structure?",
    icon: GraduationCap,
  },
  {
    category: "Faculty Directory",
    title: "Department Heads & Staff",
    description:
      "Find faculty designations, qualifications, research areas, and contact desks.",
    prompt:
      "Who is the Head of Department (HOD) and key faculty of Computer Science?",
    icon: Users,
  },
  {
    category: "Course Outcomes",
    title: "Data Structures & Core Topics",
    description:
      "Detailed unit breakdown, reference textbooks, and examination criteria.",
    prompt:
      "What are the course outcomes and syllabus modules for Data Structures in R23?",
    icon: Layers,
  },
  {
    category: "Campus & Labs",
    title: "Library, Research & Facilities",
    description:
      "Operating hours, digital library access, specialized computing labs, and centers.",
    prompt:
      "What facilities, digital resources, and timings does SRKR Central Library offer?",
    icon: Building2,
  },
];

export function HeroSuggestions({ onSelect, disabled }: HeroSuggestionsProps) {
  return (
    <div className="flex flex-col items-center justify-center py-8 md:py-14 text-center px-4 max-w-3xl mx-auto">
      {/* Magic UI Shimmer Badge */}
      <div className="mb-4">
        <ShimmerBadge>
          <Sparkles className="w-3 h-3 text-zinc-300 inline-block mr-1" />
          SRKR Intelligence • LangGraph RAG
        </ShimmerBadge>
      </div>

      {/* Hero Headline */}
      <h2 className="text-2xl sm:text-3xl md:text-4xl font-semibold tracking-tight text-zinc-100 max-w-xl">
        What would you like to explore about SRKR?
      </h2>

      {/* Subtitle */}
      <p className="mt-2 text-xs sm:text-sm text-zinc-400 max-w-md">
        Instant answers sourced directly from official college regulations,
        syllabus sheets, and department records.
      </p>

      {/* Bento Starter Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-8 w-full">
        {STARTER_CARDS.map((card, idx) => {
          const Icon = card.icon;
          return (
            <button
              key={idx}
              onClick={() => onSelect(card.prompt)}
              disabled={disabled}
              className="group relative flex flex-col justify-between p-4 rounded-2xl border border-white/[0.07] bg-zinc-950/60 hover:bg-zinc-900/50 hover:border-white/20 transition-all duration-200 text-left cursor-pointer shadow-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 group-hover:text-zinc-300 transition-colors">
                    {card.category}
                  </span>
                  <div className="w-6 h-6 rounded-lg bg-zinc-900 border border-white/[0.08] flex items-center justify-center text-zinc-400 group-hover:text-zinc-100 group-hover:border-white/20 transition-colors">
                    <ArrowUpRight className="w-3.5 h-3.5 transition-transform duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                  </div>
                </div>

                <div className="flex items-center gap-2 mb-1">
                  <Icon className="w-4 h-4 text-zinc-300 shrink-0" />
                  <h3 className="text-sm font-medium text-zinc-100 group-hover:text-white transition-colors">
                    {card.title}
                  </h3>
                </div>

                <p className="text-xs text-zinc-400 line-clamp-2 leading-relaxed">
                  {card.description}
                </p>
              </div>

              <div className="mt-3 pt-2.5 border-t border-white/[0.05] flex items-center gap-1.5 text-[11px] text-zinc-500 group-hover:text-zinc-300 transition-colors">
                <span className="font-mono truncate">“{card.prompt}”</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
