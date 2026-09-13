"use client";

import { useState, useCallback, useRef } from "react";
import { ChatMessage, AgentStep, StreamEvent } from "@/types/chat";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useAgentChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome-msg",
      role: "assistant",
      content:
        "Hello! I am your **Campus AI Academic Assistant**.\n\nI can assist you with:\n- 📚 **Syllabus & Regulations** (Curriculum structure, course codes, credits)\n- 👨‍🏫 **Faculty & Department Directory**\n- 🏢 **Campus Facilities & Labs**\n- 🎯 **Placements, Admissions & Regulations**\n\nHow can I help you today?",
      createdAt: Date.now(),
    },
  ]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (queryText: string) => {
      const trimmedQuery = queryText.trim();
      if (!trimmedQuery || isLoading) return;

      const userMessageId = `user-${Date.now()}`;
      const assistantMessageId = `asst-${Date.now()}`;

      const userMessage: ChatMessage = {
        id: userMessageId,
        role: "user",
        content: trimmedQuery,
        createdAt: Date.now(),
      };

      const initialAssistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        steps: [
          {
            node: "init",
            label: "Connecting to Campus Agent State Machine...",
            status: "active",
            timestamp: Date.now(),
          },
        ],
        sources: [],
        isStreaming: true,
        createdAt: Date.now(),
      };

      // Append user message and streaming placeholder
      setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
      setIsLoading(true);

      // Build chat history context (last 6 turns)
      const chatHistory = messages
        .filter((m) => m.id !== "welcome-msg")
        .slice(-6)
        .map((m) => ({
          role: m.role,
          content: m.content,
        }));

      abortControllerRef.current = new AbortController();

      try {
        const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: trimmedQuery,
            chat_history: chatHistory,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`Server returned HTTP ${response.status}`);
        }

        if (!response.body) {
          throw new Error("ReadableStream not supported on response");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          // Keep the last incomplete line in buffer
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine.startsWith("data:")) continue;

            const jsonStr = trimmedLine.replace(/^data:\s*/, "");
            if (!jsonStr) continue;

            try {
              const event: StreamEvent = JSON.parse(jsonStr);

              setMessages((prev) =>
                prev.map((msg) => {
                  if (msg.id !== assistantMessageId) return msg;

                  const existingSteps = [...(msg.steps || [])];

                  if (event.type === "step" && event.node && event.label) {
                    // Mark previous active step as completed
                    const updatedSteps = existingSteps.map((s) =>
                      s.status === "active" ? { ...s, status: "completed" as const } : s
                    );

                    // Add new active step
                    updatedSteps.push({
                      node: event.node,
                      label: event.label,
                      details: event.details,
                      status: "active",
                      timestamp: Date.now(),
                    });

                    return {
                      ...msg,
                      steps: updatedSteps,
                    };
                  } else if (event.type === "done") {
                    // Mark all steps completed
                    const completedSteps = existingSteps.map((s) => ({
                      ...s,
                      status: "completed" as const,
                    }));

                    return {
                      ...msg,
                      content: event.answer || "No response generated.",
                      sources: event.sources || [],
                      provider: event.provider || "Campus Advisor",
                      degraded: !!event.degraded,
                      isStreaming: false,
                      steps: completedSteps,
                    };
                  } else if (event.type === "error") {
                    return {
                      ...msg,
                      content: `⚠️ **Agent Error:** ${event.message || "An unexpected error occurred."}`,
                      isStreaming: false,
                      steps: existingSteps.map((s) => ({
                        ...s,
                        status: "failed" as const,
                      })),
                    };
                  }

                  return msg;
                })
              );
            } catch {
              // Ignore partial JSON parse errors in stream chunk
            }
          }
        }
      } catch (err: any) {
        if (err.name === "AbortError") return;

        setMessages((prev) =>
          prev.map((msg) => {
            if (msg.id !== assistantMessageId) return msg;
            return {
              ...msg,
              content: `⚠️ **Network Error**: Unable to connect to backend server at \`${API_BASE_URL}\`. Please ensure the FastAPI backend is running.`,
              isStreaming: false,
            };
          })
        );
      } finally {
        setIsLoading(false);
        abortControllerRef.current = null;
      }
    },
    [isLoading, messages]
  );

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
    }
  }, []);

  const clearChat = useCallback(() => {
    setMessages([
      {
        id: "welcome-msg",
        role: "assistant",
        content:
          "Conversation reset. How can I assist you with college academic information?",
        createdAt: Date.now(),
      },
    ]);
  }, []);

  return {
    messages,
    isLoading,
    sendMessage,
    stopGeneration,
    clearChat,
  };
}
