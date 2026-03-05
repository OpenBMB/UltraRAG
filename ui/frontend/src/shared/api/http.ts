import type { ApiErrorPayload } from "@/shared/api/types";

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

type RequestOptions = {
  method?: HttpMethod;
  body?: unknown;
  headers?: HeadersInit;
  signal?: AbortSignal;
};

const DEFAULT_HEADERS: HeadersInit = {
  "Content-Type": "application/json",
};

export function buildApiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  const apiBase = (import.meta.env.VITE_API_BASE ?? "").trim();
  if (!apiBase) return path;
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

async function parseResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }
  return (await response.text()) as unknown as T;
}

export class ApiHttpError extends Error {
  status: number;
  payload: ApiErrorPayload | null;

  constructor(status: number, payload: ApiErrorPayload | null, fallbackMessage: string) {
    super(payload?.error ?? fallbackMessage);
    this.name = "ApiHttpError";
    this.status = status;
    this.payload = payload;
  }
}

export async function httpRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, headers, signal } = options;
  const response = await fetch(buildApiUrl(path), {
    method,
    credentials: "include",
    headers: {
      ...DEFAULT_HEADERS,
      ...headers,
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    let payload: ApiErrorPayload | null = null;
    try {
      payload = await parseResponse<ApiErrorPayload>(response);
    } catch {
      payload = null;
    }
    throw new ApiHttpError(response.status, payload, `Request failed: ${method} ${path}`);
  }

  return parseResponse<T>(response);
}
