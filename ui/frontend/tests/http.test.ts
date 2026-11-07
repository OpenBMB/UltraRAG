import assert from "node:assert/strict";
import test from "node:test";
import { ApiHttpError, httpRequest } from "../src/shared/api/http.ts";

test("request timeout aborts a stalled request without retrying it", async () => {
  const original = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async (_input, options) => {
    calls += 1;
    return new Promise((_resolve, reject) => {
      options?.signal?.addEventListener("abort", () => reject(options.signal?.reason), { once: true });
    });
  };
  try {
    await assert.rejects(httpRequest("https://test.invalid", { method: "POST", timeoutMs: 5 }), { name: "TimeoutError" });
    assert.equal(calls, 1);
  } finally { globalThis.fetch = original; }
});

test("caller cancellation keeps its original reason", async () => {
  const original = globalThis.fetch;
  const controller = new AbortController();
  controller.abort(new Error("user cancelled"));
  globalThis.fetch = async (_input, options) => { throw options?.signal?.reason; };
  try {
    await assert.rejects(httpRequest("https://test.invalid", { signal: controller.signal }), /user cancelled/);
  } finally { globalThis.fetch = original; }
});

test("the timeout remains active while reading a stalled JSON body", { timeout: 500 }, async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async (_input, options) => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('{"incomplete":'));
      options?.signal?.addEventListener("abort", () => controller.error(options.signal?.reason), { once: true });
    },
  }), { headers: { "Content-Type": "application/json" } });
  try {
    await assert.rejects(httpRequest("https://test.invalid", { timeoutMs: 5 }), { name: "TimeoutError" });
  } finally { globalThis.fetch = original; }
});

test("Headers objects preserve custom headers and HTTP errors retain their status", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async (_input, options) => {
    const headers = new Headers(options?.headers);
    assert.equal(headers.get("X-Test"), "ok");
    assert.equal(headers.get("Content-Type"), "application/json");
    return new Response(JSON.stringify({ error: "rate limited" }), { status: 429, headers: { "Content-Type": "application/json" } });
  };
  try {
    await assert.rejects(httpRequest("https://test.invalid", { headers: new Headers({ "X-Test": "ok" }) }), (error) => error instanceof ApiHttpError && error.status === 429 && error.message === "rate limited");
  } finally { globalThis.fetch = original; }
});
