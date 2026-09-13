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
  Check,
  Loader2,
} from "lucide-react";

interface ReasoningStepsProps {
  steps: AgentStep[];
  isStreaming?: boolean;
}

function getFriendlyPhase(node: string) {
  switch (node) {
    case "cache":
    case "cache_check":
      return {
        stage: "Checking Cache",
        detail: "Checking quick verified answers",
        icon: Brain,
      };
    case "planner":
      return {
        stage: "Query Analysis",
        detail: "Formulating multi-angle search query",
        icon: Brain,
      };
    case "executor":
    case "reranker":
      return {
        stage: "Searching Records",
        detail: "Searching curriculum & regulations",
        icon: Search,
      };
    case "validator":
    case "guard":
      return {
        stage: "Verifying Facts",
        detail: "Validating against official college guidelines",
        icon: ShieldCheck,
      };
    case "generator":
    case "responder":
      return {
        stage: "Composing Answer",
        detail: "Synthesizing response with citations",
        icon: Sparkles,
      };
    default:
      return {
        stage: "Thinking",
        detail: "Processing your request",
        icon: Brain,
      };
  }
}

export function ReasoningSteps({ steps, isStreaming }: ReasoningStepsProps) {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  if (!steps || steps.length === 0) return null;

  const activeStep = steps.find((s) => s.status === "active");
  const activePhase = activeStep ? getFriendlyPhase(activeStep.node) : null;

  const consolidatedStages = [
    {
      key: "planning",
      title: "Question Understanding & Planning",
      desc: "Decomposed query and formulated optimal dense & keyword search strategy",
      isDone: steps.some((s) => ["planner", "cache"].includes(s.node)),
      isActive: activeStep
        ? ["planner", "cache", "init"].includes(activeStep.node)
        : false,
    },
    {
      key: "searching",
      title: "Campus Knowledge Base Retrieval",
      desc: "Retrieved relevant official syllabus documents, faculty rosters, and regulations",
      isDone: steps.some(
        (s) =>
          ["executor", "reranker"].includes(s.node) && s.status === "completed"
      ),
      isActive: activeStep
        ? ["executor", "reranker"].includes(activeStep.node)
        : false,
    },
    {
      key: "verifying",
      title: "Fact & Policy Verification",
      desc: "Cross-checked course codes, credit distribution, and source accuracy",
      isDone: steps.some(
        (s) =>
          ["validator", "guard"].includes(s.node) && s.status === "completed"
      ),
      isActive: activeStep
        ? ["validator", "guard"].includes(activeStep.node)
        : false,
    },
    {
      key: "generating",
      title: "Synthesizing Response",
      desc: "Generated coherent response with cited references",
      isDone: !isStreaming && steps.length > 0,
      isActive: activeStep
        ? ["generator", "responder"].includes(activeStep.node)
        : false,
    },
  ];

  return (
    <div className="mb-3 w-full">
      {/* Status Pill Button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`flex items-center justify-between w-full px-3 py-1.5 text-xs font-medium rounded-xl border transition-all duration-200 cursor-pointer ${
          isStreaming
            ? "bg-rose-50/80 border-rose-200 text-[#800020] shadow-xs"
            : "bg-slate-50 border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-100"
        }`}
      >
        <div className="flex items-center gap-2 truncate">
          {isStreaming ? (
            <>
              <Loader2 className="w-3.5 h-3.5 text-[#800020] animate-spin shrink-0" />
              <span className="font-semibold text-[#800020]">
                {activePhase ? `${activePhase.stage}...` : "Thinking..."}
              </span>
              <span className="text-slate-500 text-[11px] truncate hidden sm:inline">
                ({activePhase?.detail})
              </span>
            </>
          ) : (
            <>
              <div className="w-4 h-4 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center shrink-0">
                <Check className="w-2.5 h-2.5 stroke-[3]" />
              </div>
              <span className="text-slate-700 font-medium text-[11px]">
                Agent Reasoned in {steps.length} Steps
              </span>
            </>
          )}
        </div>

        <div className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-800 shrink-0 ml-2">
          <span>{isExpanded ? "Hide" : "Details"}</span>
          {isExpanded ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )}
        </div>
      </button>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="mt-2 p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-2.5">
          {consolidatedStages.map((stage) => {
            if (!stage.isDone && !stage.isActive) return null;

            return (
              <div key={stage.key} className="flex items-start gap-2.5 text-xs">
                <div className="mt-0.5 shrink-0">
                  {stage.isActive ? (
                    <Loader2 className="w-3.5 h-3.5 text-[#800020] animate-spin" />
                  ) : stage.isDone ? (
                    <div className="w-3.5 h-3.5 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center">
                      <Check className="w-2 h-2 stroke-[3]" />
                    </div>
                  ) : (
                    <div className="w-3.5 h-3.5 rounded-full border border-slate-300" />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div
                    className={`font-semibold ${
                      stage.isActive ? "text-[#800020]" : "text-slate-800"
                    }`}
                  >
                    {stage.title}
                  </div>
                  <div className="text-[11px] text-slate-500 leading-normal mt-0.5">
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
