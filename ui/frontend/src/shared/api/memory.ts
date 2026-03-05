import { httpRequest } from "@/shared/api/http";

type MemoryResponse = {
  user_id: string;
  path: string;
  content: string;
};

function memoryPath(userId?: string): string {
  return userId ? `/api/memory/${encodeURIComponent(userId)}` : "/api/memory";
}

export async function fetchMemory(userId?: string): Promise<MemoryResponse> {
  return httpRequest<MemoryResponse>(memoryPath(userId));
}

export async function saveMemory(content: string, userId?: string): Promise<{ status: string }> {
  return httpRequest<{ status: string }>(memoryPath(userId), {
    method: "PUT",
    body: { content },
  });
}
