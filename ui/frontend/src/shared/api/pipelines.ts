import { httpRequest } from "@/shared/api/http";
import type {
  PipelineChatHistoryResponse,
  PipelineItem,
} from "@/shared/api/types";

export type ServerToolItem = {
  id?: string;
  server: string;
  tool: string;
  kind?: string;
  input?: unknown;
  output?: unknown;
};

export async function fetchPipelines(): Promise<PipelineItem[]> {
  const payload = await httpRequest<unknown>("/api/pipelines");
  return Array.isArray(payload) ? (payload as PipelineItem[]) : [];
}

export async function fetchServerTools(): Promise<ServerToolItem[]> {
  const payload = await httpRequest<unknown>("/api/tools");
  return Array.isArray(payload) ? (payload as ServerToolItem[]) : [];
}

export async function startPipelineDemoSession(name: string, sessionId: string): Promise<unknown> {
  return httpRequest<unknown>(`/api/pipelines/${encodeURIComponent(name)}/demo/start`, {
    method: "POST",
    body: { session_id: sessionId },
  });
}

export async function stopPipelineDemoSession(sessionId: string): Promise<unknown> {
  return httpRequest<unknown>("/api/pipelines/demo/stop", {
    method: "POST",
    body: { session_id: sessionId },
  });
}

export async function fetchPipelineChatHistory(sessionId: string): Promise<PipelineChatHistoryResponse> {
  return httpRequest<PipelineChatHistoryResponse>(
    `/api/pipelines/chat/history?session_id=${encodeURIComponent(sessionId)}`,
  );
}

export async function stopPipelineChatGeneration(sessionId: string): Promise<unknown> {
  return httpRequest<unknown>("/api/pipelines/chat/stop", {
    method: "POST",
    body: { session_id: sessionId },
  });
}

export async function clearPipelineChatHistory(sessionId: string): Promise<{ status: string }> {
  return httpRequest<{ status: string }>("/api/pipelines/chat/clear-history", {
    method: "POST",
    body: { session_id: sessionId },
  });
}

type BackgroundChatPayload = {
  question: string;
  sessionId: string;
  userId: string;
  collectionName?: string;
};

export async function startPipelineBackgroundChat(
  name: string,
  payload: BackgroundChatPayload,
): Promise<{ status: string; task_id: string; message?: string }> {
  const dynamicParams: Record<string, unknown> = {
    memory: { user_id: payload.userId },
  };
  if (payload.collectionName) {
    dynamicParams.collection_name = payload.collectionName;
  }
  return httpRequest<{ status: string; task_id: string; message?: string }>(
    `/api/pipelines/${encodeURIComponent(name)}/chat/background`,
    {
      method: "POST",
      body: {
        question: payload.question,
        session_id: payload.sessionId,
        dynamic_params: dynamicParams,
      },
    },
  );
}

export async function fetchPipelineConfig(name: string): Promise<Record<string, unknown>> {
  return httpRequest<Record<string, unknown>>(`/api/pipelines/${encodeURIComponent(name)}`);
}

export async function savePipelineYaml(name: string, yamlText: string): Promise<unknown> {
  const response = await fetch(`/api/pipelines/${encodeURIComponent(name)}/yaml`, {
    method: "PUT",
    credentials: "include",
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
    body: yamlText,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function buildPipeline(name: string): Promise<unknown> {
  return httpRequest<unknown>(`/api/pipelines/${encodeURIComponent(name)}/build`, {
    method: "POST",
  });
}

export async function fetchPipelineParameters(name: string): Promise<Record<string, unknown>> {
  return httpRequest<Record<string, unknown>>(
    `/api/pipelines/${encodeURIComponent(name)}/parameters`,
  );
}

export async function savePipelineParameters(
  name: string,
  params: Record<string, unknown>,
): Promise<unknown> {
  return httpRequest<unknown>(`/api/pipelines/${encodeURIComponent(name)}/parameters`, {
    method: "PUT",
    body: params,
  });
}

export async function createPipeline(name: string, steps: unknown[] = []): Promise<{
  name: string;
}> {
  return httpRequest<{ name: string }>("/api/pipelines", {
    method: "POST",
    body: { name, pipeline: steps },
  });
}

export async function deletePipeline(name: string): Promise<{ status: string }> {
  return httpRequest<{ status: string }>(`/api/pipelines/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export async function renamePipeline(name: string, newName: string): Promise<{
  status: string;
  new_name?: string;
}> {
  return httpRequest<{ status: string; new_name?: string }>(
    `/api/pipelines/${encodeURIComponent(name)}/rename`,
    {
      method: "POST",
      body: { new_name: newName },
    },
  );
}

export async function parsePipelineYaml(yamlText: string): Promise<Record<string, unknown>> {
  const response = await fetch("/api/pipelines/parse", {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
    body: yamlText,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Parse pipeline failed (${response.status})`);
  }
  return response.json();
}
