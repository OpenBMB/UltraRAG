import assert from "node:assert/strict";
import test from "node:test";
import { readJsonEvents } from "../src/shared/api/sse.ts";

async function decode(text: string, chunkSize: number) {
  const bytes = new TextEncoder().encode(text);
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let i = 0; i < bytes.length; i += chunkSize) controller.enqueue(bytes.slice(i, i + chunkSize));
      controller.close();
    },
  });
  const values = [];
  for await (const value of readJsonEvents(body)) values.push(value);
  assert.equal(body.locked, false);
  return values;
}

for (const newline of ["\n", "\r\n", "\r"]) {
  test(`decodes ${JSON.stringify(newline)} frames even at byte boundaries`, async () => {
    const frames = [
      ': heartbeat', '',
      'data: {"type":"token",', 'data: "content":"你好 🧪"}', '',
      'data: invalid JSON', '',
      'data: {"type":"final","data":{"answer":"done"}}', '',
      'data: [DONE]', '', '',
    ].join(newline);
    for (const size of [1, 2, 3, 7, 1024]) {
      assert.deepEqual(await decode(frames, size), [
        { type: "token", content: "你好 🧪" },
        { type: "final", data: { answer: "done" } },
      ]);
    }
  });
}

test("accepts a final JSON frame when the backend closes without a separator", async () => {
  assert.deepEqual(await decode('data: {"type":"final"}', 1), [{ type: "final" }]);
});

test("supports mixed line endings in the same stream", async () => {
  assert.deepEqual(await decode('data: {"type":"one"}\n\r\ndata: {"type":"two"}\r\n\r', 1), [
    { type: "one" }, { type: "two" },
  ]);
});

test("cancels the underlying reader and releases its lock on early exit", async () => {
  let cancelled = false;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('data: {"type":"token"}\n\n'));
    },
    cancel() { cancelled = true; },
  });
  for await (const value of readJsonEvents(body)) {
    assert.deepEqual(value, { type: "token" });
    break;
  }
  assert.equal(cancelled, true);
  assert.equal(body.locked, false);
});
