import type { ApiErrorPayload } from "@/shared/api/types";

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

type RequestOptions = {
  method?: HttpMethod;
  body?: unknown;
  headers?: HeadersInit;
  signal?: AbortSignal;
  timeoutMs?: number;
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
  const { method = "GET", body, headers, signal, timeoutMs = 120_000 } = options;
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal?.reason);
  if (signal?.aborted) onAbort();
  else signal?.addEventListener("abort", onAbort, { once: true });
  const timeout = timeoutMs > 0
    ? setTimeout(() => controller.abort(new DOMException("Request timed out. Please try again.", "TimeoutError")), timeoutMs)
    : undefined;
  try {
    const requestHeaders = new Headers(DEFAULT_HEADERS);
    new Headers(headers).forEach((value, key) => requestHeaders.set(key, value));
    const response = await fetch(buildApiUrl(path), {
      method,
      credentials: "include",
      headers: requestHeaders,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
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
    return await parseResponse<T>(response);
  } catch (error) {
    if (controller.signal.aborted) throw controller.signal.reason;
    throw error;
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
    signal?.removeEventListener("abort", onAbort);
  }
}
