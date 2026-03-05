import { buildApiUrl, httpRequest } from "@/shared/api/http";
import type { ChatMessage, ChatSession, SourceDoc } from "@/shared/api/types";
import { createClientId } from "@/shared/lib/id";

type UpsertChatSessionPayload = {
  id: string;
  title?: string;
  pipeline?: string | null;
  messages?: ChatMessage[];
  timestamp?: number;
};

export async function fetchChatSessions(limit = 300): Promise<ChatSession[]> {
  const payload = await httpRequest<unknown>(`/api/chat/sessions?limit=${limit}`);
  return Array.isArray(payload) ? (payload as ChatSession[]) : [];
}

export async function fetchChatSession(sessionId: string): Promise<ChatSession> {
  return httpRequest<ChatSession>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`);
}

export async function upsertChatSession(payload: UpsertChatSessionPayload): Promise<ChatSession> {
  return httpRequest<ChatSession>("/api/chat/sessions", {
    method: "POST",
    body: payload,
  });
}

export async function renameChatSession(sessionId: string, title: string): Promise<ChatSession> {
  return httpRequest<ChatSession>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PUT",
    body: { title },
  });
}

export async function createChatSession(title = "New Chat"): Promise<ChatSession> {
  return upsertChatSession({
    id: createClientId(),
    title,
    timestamp: Date.now(),
    messages: [],
  });
}

export async function deleteChatSession(sessionId: string): Promise<{ status: string }> {
  return httpRequest<{ status: string }>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

export async function clearChatSessions(): Promise<{ status: string; count: number }> {
  return httpRequest<{ status: string; count: number }>("/api/chat/sessions", {
    method: "DELETE",
  });
}

type ExportChatDocxPayload = {
  text: string;
  question?: string;
  sources?: SourceDoc[];
};

function parseFilename(contentDisposition: string | null): string {
  if (!contentDisposition) return "ultrarag-chat.docx";
  const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1]).replaceAll(/["']/g, "");
    } catch {
      return utf8Match[1].replaceAll(/["']/g, "");
    }
  }
  const plainMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
  if (plainMatch?.[1]) return plainMatch[1];
  return "ultrarag-chat.docx";
}

export async function exportChatDocx(payload: ExportChatDocxPayload): Promise<{
  blob: Blob;
  filename: string;
}> {
  const response = await fetch(buildApiUrl("/api/chat/export/docx"), {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: payload.text,
      question: payload.question ?? "",
      sources: payload.sources ?? [],
    }),
  });
  if (!response.ok) {
    let message = response.statusText || "Export docx failed";
    try {
      const json = (await response.json()) as { error?: string };
      message = json.error ?? message;
    } catch {
      // no-op
    }
    throw new Error(message);
  }
  const blob = await response.blob();
  return {
    blob,
    filename: parseFilename(response.headers.get("Content-Disposition")),
  };
}
