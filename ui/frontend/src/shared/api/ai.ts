import { httpRequest } from "@/shared/api/http";

export type AiConnectionPayload = {
  provider: string;
  baseUrl: string;
  apiKey: string;
  model: string;
};

export type AiChatPayload = {
  settings: AiConnectionPayload;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  context?: Record<string, unknown>;
  stream?: boolean;
};

export async function testAiConnection(payload: AiConnectionPayload): Promise<{
  success: boolean;
  model?: string;
  error?: string;
}> {
  return httpRequest<{ success: boolean; model?: string; error?: string }>("/api/ai/test", {
    method: "POST",
    body: payload,
  });
}

export async function chatWithAi(payload: AiChatPayload): Promise<{
  content?: string;
  message?: string;
  answer?: string;
  actions?: unknown[];
  error?: string;
  [key: string]: unknown;
}> {
  return httpRequest<{
    content?: string;
    message?: string;
    answer?: string;
    actions?: unknown[];
    error?: string;
    [key: string]: unknown;
  }>("/api/ai/chat", {
    method: "POST",
    body: payload,
  });
}
