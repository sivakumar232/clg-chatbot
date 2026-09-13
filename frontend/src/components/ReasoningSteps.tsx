"use client";

import { useState } from "react";
import { AgentStep } from "@/types/chat";
import {
  ChevronDown,
  ChevronUp,
  Brain,
  Search,
  ShieldCheck,
  Sparkles,
  CheckCircle2,
  Loader2,
} from "lucide-react";

interface ReasoningStepsProps {
  steps: AgentStep[];
  isStreaming?: boolean;
}

// Convert technical node names into clean, simple user-facing stages
function getFriendlyPhase(node: string) {
  switch (node) {
    case "cache":
    case "cache_check":
      return {
        stage: "Checking Memory",
        simpleLabel: "Checking quick answers",
        icon: Brain,
        color: "text-indigo-400",
      };
    case "planner":
      return {
        stage: "Planning",
        simpleLabel: "Understanding question & planning search",
        icon: Brain,
        color: "text-indigo-400",
      };
    case "executor":
    case "reranker":
      return {
        stage: "Searching",
        simpleLabel: "Searching SRKR syllabus & department records",
        icon: Search,
        color: "text-cyan-400",
      };
    case "validator":
    case "guard":
      return {
        stage: "Verifying",
        simpleLabel: "Verifying accuracy & official regulations",
        icon: ShieldCheck,
        color: "text-emerald-400",
      };
    case "generator":
    case "responder":
      return {
        stage: "Generating",
        simpleLabel: "Writing answer with citations",
        icon: Sparkles,
        color: "text-amber-400",
      };
    default:
      return {
        stage: "Thinking",
        simpleLabel: "Processing request",
        icon: Brain,
        color: "text-slate-400",
      };
  }
}

export function ReasoningSteps({ steps, isStreaming }: ReasoningStepsProps) {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  if (!steps || steps.length === 0) return null;

  const activeStep = steps.find((s) => s.status === "active");
  const activePhase = activeStep ? getFriendlyPhase(activeStep.node) : null;
  const ActiveIcon = activePhase?.icon || Loader2;

  // Consolidate unique stages for clean, simple bullet display
  const consolidatedStages = [
    {
      key: "planning",
      title: "Understanding & Planning",
      desc: "Analyzed query intent and selected optimal search strategy",
      isDone: steps.some((s) => ["planner", "cache"].includes(s.node)),
      isActive: activeStep ? ["planner", "cache", "init"].includes(activeStep.node) : false,
    },
    {
      key: "searching",
      title: "Searching College Knowledge Base",
      desc: "Retrieved and ranked relevant SRKR syllabus and college documents",
      isDone: steps.some((s) => ["executor", "reranker"].includes(s.node) && s.status === "completed"),
      isActive: activeStep ? ["executor", "reranker"].includes(activeStep.node) : false,
    },
    {
      key: "verifying",
      title: "Verifying Information",
      desc: "Checked syllabus regulations, course codes, and source facts",
      isDone: steps.some((s) => ["validator", "guard"].includes(s.node) && s.status === "completed"),
      isActive: activeStep ? ["validator", "guard"].includes(activeStep.node) : false,
    },
    {
      key: "generating",
      title: "Generating Response",
      desc: "Synthesizing answer with verified citations",
      isDone: !isStreaming && steps.length > 0,
      isActive: activeStep ? ["generator", "responder"].includes(activeStep.node) : false,
    },
  ];

  return (
    <div className="mb-3 w-full max-w-xl">
      {/* Sleek Minimal Status Pill */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`flex items-center justify-between w-full px-3 py-1.5 text-xs font-medium rounded-xl border transition-all duration-200 ${
          isStreaming
            ? "bg-indigo-950/30 border-indigo-500/30 text-indigo-200 shadow-sm"
            : "bg-slate-900/50 border-slate-800/80 text-slate-400 hover:text-slate-200 hover:border-slate-700"
        }`}
      >
        <div className="flex items-center gap-2 truncate">
          {isStreaming ? (
            <>
              <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin shrink-0" />
              <span className="font-medium text-indigo-300">
                {activePhase ? `${activePhase.stage}...` : "Thinking..."}
              </span>
              <span className="text-slate-400 text-[11px] truncate hidden sm:inline">
                ({activePhase?.simpleLabel})
              </span>
            </>
          ) : (
            <>
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              <span className="text-slate-300">Thought Process Completed</span>
            </>
          )}
        </div>

        <div className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300 shrink-0 ml-2">
          <span>{isExpanded ? "Hide" : "Details"}</span>
          {isExpanded ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )}
        </div>
      </button>

      {/* Expanded Simple View */}
      {isExpanded && (
        <div className="mt-2 p-3 bg-slate-950/70 rounded-xl border border-slate-800/80 space-y-2.5 backdrop-blur-sm shadow-inner">
          {consolidatedStages.map((stage) => {
            if (!stage.isDone && !stage.isActive) return null;

            return (
              <div
                key={stage.key}
                className="flex items-start gap-2.5 text-xs"
              >
                <div className="mt-0.5 shrink-0">
                  {stage.isActive ? (
                    <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
                  ) : stage.isDone ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <div className="w-3.5 h-3.5 rounded-full border border-slate-700" />
                  )}
                </div>

                <div>
                  <div className={`font-medium ${stage.isActive ? "text-indigo-300 font-semibold" : "text-slate-200"}`}>
                    {stage.title}
                  </div>
                  <div className="text-[11px] text-slate-400 leading-normal">
                    {stage.desc}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
