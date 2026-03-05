import { buildApiUrl, ApiHttpError } from "@/shared/api/http";
import type { PipelineChatEvent } from "@/shared/api/types";

type StreamPipelineChatPayload = {
  pipelineName: string;
  question: string;
  sessionId: string;
  chatSessionId: string;
  history: Array<{ role: "user" | "assistant"; text: string }>;
  userId: string;
  collectionName?: string;
  signal?: AbortSignal;
};

function toEvent(raw: unknown): PipelineChatEvent | null {
  if (!raw || typeof raw !== "object") return null;
  const eventLike = raw as Record<string, unknown>;
  if (typeof eventLike.type !== "string") return null;
  return eventLike as PipelineChatEvent;
}

export async function* streamPipelineChat(
  payload: StreamPipelineChatPayload,
): AsyncGenerator<PipelineChatEvent> {
  const endpoint = buildApiUrl(`/api/pipelines/${encodeURIComponent(payload.pipelineName)}/chat`);
  const dynamicParams: Record<string, unknown> = {
    memory: { user_id: payload.userId },
  };
  if (payload.collectionName) {
    dynamicParams.collection_name = payload.collectionName;
  }

  const response = await fetch(endpoint, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: payload.question,
      history: payload.history,
      is_demo: true,
      session_id: payload.sessionId,
      chat_session_id: payload.chatSessionId,
      dynamic_params: dynamicParams,
    }),
    signal: payload.signal,
  });

  if (!response.ok) {
    let apiMessage = response.statusText;
    try {
      const errorPayload = (await response.json()) as { error?: string; details?: string };
      apiMessage = errorPayload.error ?? errorPayload.details ?? apiMessage;
    } catch {
      // noop
    }
    throw new ApiHttpError(response.status, { error: apiMessage }, "Pipeline chat request failed");
  }

  if (!response.body) {
    throw new Error("Pipeline chat stream body is empty");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const line = block
        .split("\n")
        .map((segment) => segment.trim())
        .find((segment) => segment.startsWith("data:"));
      if (!line) continue;
      const jsonPayload = line.slice(5).trim();
      if (!jsonPayload) continue;
      try {
        const parsed = JSON.parse(jsonPayload);
        const event = toEvent(parsed);
        if (event) yield event;
      } catch {
        // Skip malformed event chunk.
      }
    }
  }
}
