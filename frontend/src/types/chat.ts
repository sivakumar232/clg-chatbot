export type AgentNodeName =
  | "cache_check"
  | "planner"
  | "executor"
  | "reranker"
  | "validator"
  | "generator"
  | "guard"
  | "responder"
  | "cache_write";

export interface AgentStep {
  node: AgentNodeName | string;
  label: string;
  details?: Record<string, any>;
  status: "active" | "completed" | "failed";
  timestamp: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  steps?: AgentStep[];
  sources?: string[];
  provider?: string;
  cacheHit?: boolean;
  degraded?: boolean;
  isStreaming?: boolean;
  createdAt: number;
}

export interface StreamEvent {
  type: "start" | "step" | "done" | "error";
  query?: string;
  node?: string;
  label?: string;
  details?: Record<string, any>;
  answer?: string;
  sources?: string[];
  provider?: string;
  degraded?: boolean;
  message?: string;
}
