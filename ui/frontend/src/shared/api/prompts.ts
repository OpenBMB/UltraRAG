import { httpRequest } from "@/shared/api/http";

export type PromptFileMeta = {
  name: string;
  path: string;
  size: number;
};

function encodePromptPath(path: string): string {
  return path
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

export async function fetchPrompts(): Promise<PromptFileMeta[]> {
  const payload = await httpRequest<unknown>("/api/prompts");
  return Array.isArray(payload) ? (payload as PromptFileMeta[]) : [];
}

export async function fetchPromptContent(path: string): Promise<string> {
  const payload = await httpRequest<{ path: string; content: string }>(
    `/api/prompts/${encodePromptPath(path)}`,
  );
  return typeof payload.content === "string" ? payload.content : "";
}

export async function savePromptContent(path: string, content: string): Promise<unknown> {
  return httpRequest<unknown>(`/api/prompts/${encodePromptPath(path)}`, {
    method: "PUT",
    body: { content },
  });
}

export async function createPrompt(path: string, content = ""): Promise<unknown> {
  return httpRequest<unknown>("/api/prompts", {
    method: "POST",
    body: {
      name: path,
      content,
    },
  });
}

export async function deletePrompt(path: string): Promise<unknown> {
  return httpRequest<unknown>(`/api/prompts/${encodePromptPath(path)}`, {
    method: "DELETE",
  });
}

export async function renamePrompt(path: string, newPath: string): Promise<unknown> {
  return httpRequest<unknown>(`/api/prompts/${encodePromptPath(path)}/rename`, {
    method: "POST",
    body: {
      new_name: newPath,
    },
  });
}
