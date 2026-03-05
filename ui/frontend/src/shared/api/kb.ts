import { httpRequest } from "@/shared/api/http";

export type KbFileEntry = {
  name: string;
  display_name?: string;
  path?: string;
  size?: number;
  mtime?: number;
  category: string;
  type?: string;
  count?: number;
  file_count?: number;
};

export type KbFilesResponse = {
  raw: KbFileEntry[];
  corpus: KbFileEntry[];
  chunks: KbFileEntry[];
  index: KbFileEntry[];
  db_status?: string;
  db_config?: Record<string, unknown>;
};

export type KbVisibilityPayload = {
  collection_name: string;
  owner_user_id?: string;
  visibility: "private" | "public" | "shared";
  visible_users: string[];
  can_manage?: boolean;
  can_view?: boolean;
};

export async function fetchKbFiles(): Promise<KbFilesResponse> {
  return httpRequest<KbFilesResponse>("/api/kb/files");
}

export async function uploadKbFiles(files: File[]): Promise<unknown> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("file", file);
  }
  const response = await fetch("/api/kb/upload", {
    method: "POST",
    credentials: "include",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function deleteKbFile(category: string, filename: string): Promise<unknown> {
  return httpRequest<unknown>(`/api/kb/files/${encodeURIComponent(category)}/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });
}

export async function syncMemoryToKb(indexMode: "append" | "overwrite" = "append"): Promise<{
  task_id: string;
  status: string;
}> {
  return httpRequest<{ task_id: string; status: string }>("/api/kb/sync-memory", {
    method: "POST",
    body: { index_mode: indexMode },
  });
}

export async function clearMemoryVectors(): Promise<{
  status: string;
  cleared_count?: number;
  collection_name?: string;
}> {
  return httpRequest<{
    status: string;
    cleared_count?: number;
    collection_name?: string;
  }>("/api/kb/clear-memory", {
    method: "POST",
    body: {},
  });
}

export async function fetchKbTaskStatus(taskId: string): Promise<Record<string, unknown>> {
  return httpRequest<Record<string, unknown>>(`/api/kb/status/${encodeURIComponent(taskId)}`);
}

export async function fetchKbConfig(): Promise<Record<string, unknown>> {
  return httpRequest<Record<string, unknown>>("/api/kb/config");
}

export async function saveKbConfig(payload: Record<string, unknown>): Promise<{ status: string }> {
  return httpRequest<{ status: string }>("/api/kb/config", {
    method: "POST",
    body: payload,
  });
}

export async function listKbVisibilityUsers(): Promise<{ users: string[] }> {
  return httpRequest<{ users: string[] }>("/api/kb/visibility/users");
}

export async function fetchKbVisibility(collectionName: string): Promise<KbVisibilityPayload> {
  return httpRequest<KbVisibilityPayload>(`/api/kb/visibility/${encodeURIComponent(collectionName)}`);
}

export async function saveKbVisibility(
  collectionName: string,
  payload: {
    visibility: "private" | "public" | "shared";
    visible_users?: string[];
  },
): Promise<KbVisibilityPayload> {
  return httpRequest<KbVisibilityPayload>(`/api/kb/visibility/${encodeURIComponent(collectionName)}`, {
    method: "POST",
    body: payload,
  });
}

export async function inspectKbFolder(
  category: "raw" | "corpus" | "chunks",
  name: string,
): Promise<{ files: Array<{ name: string; size: number }> }> {
  const query = `category=${encodeURIComponent(category)}&name=${encodeURIComponent(name)}`;
  return httpRequest<{ files: Array<{ name: string; size: number }> }>(`/api/kb/files/inspect?${query}`);
}

export async function clearKbStaging(): Promise<Record<string, unknown>> {
  return httpRequest<Record<string, unknown>>("/api/kb/staging/clear", {
    method: "POST",
    body: {},
  });
}

export async function runKbTask(payload: {
  pipeline_name: string;
  target_file: string;
  collection_name?: string;
  index_mode?: "new" | "append" | "overwrite";
  chunk_backend?: string;
  tokenizer_or_token_counter?: string;
  chunk_size?: number;
  use_title?: boolean;
  emb_api_key?: string;
  emb_base_url?: string;
  emb_model_name?: string;
}): Promise<{ status: string; task_id: string; message?: string }> {
  return httpRequest<{ status: string; task_id: string; message?: string }>("/api/kb/run", {
    method: "POST",
    body: payload,
  });
}
