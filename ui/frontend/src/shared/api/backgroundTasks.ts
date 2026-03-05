import { httpRequest } from "@/shared/api/http";

export type BackgroundTask = {
  task_id: string;
  pipeline_name: string;
  status: string;
  question?: string;
  full_question?: string;
  created_at?: number;
  completed_at?: number | null;
  result?: string | null;
  result_preview?: string | null;
  error?: string | null;
  sources?: unknown[];
  [key: string]: unknown;
};

export async function fetchBackgroundTasks(limit = 20): Promise<BackgroundTask[]> {
  const payload = await httpRequest<unknown>(`/api/background-tasks?limit=${limit}`);
  return Array.isArray(payload) ? (payload as BackgroundTask[]) : [];
}

export async function fetchBackgroundTask(taskId: string): Promise<BackgroundTask> {
  return httpRequest<BackgroundTask>(`/api/background-tasks/${encodeURIComponent(taskId)}`);
}

export async function deleteBackgroundTask(taskId: string): Promise<{ status: string }> {
  return httpRequest<{ status: string }>(`/api/background-tasks/${encodeURIComponent(taskId)}`, {
    method: "DELETE",
  });
}

export async function clearCompletedBackgroundTasks(): Promise<{ status: string; count: number }> {
  return httpRequest<{ status: string; count: number }>("/api/background-tasks/clear-completed", {
    method: "POST",
    body: {},
  });
}
