import assert from "node:assert/strict";
import test from "node:test";
import { ChatTurnController } from "../src/features/chat/ChatTurnController.ts";

test("a stopped turn cannot mutate a new turn or finish while its server is stopping", async () => {
  const turns = new ChatTurnController();
  const first = turns.begin();
  let resolveStop!: () => void;
  const stop = turns.stop(() => new Promise<void>((resolve) => { resolveStop = resolve; }));
  assert.equal(first.controller.signal.aborted, true);
  assert.equal(turns.isCurrent(first), false);
  turns.finish(first);
  assert.throws(() => turns.begin(), /already active/);
  await Promise.resolve();
  resolveStop();
  await stop;
  const second = turns.begin();
  turns.finish(first);
  assert.equal(turns.isCurrent(second), true);
  turns.finish(second);
  assert.equal(turns.busy, false);
});

test("concurrent stop requests share one server stop", async () => {
  const turns = new ChatTurnController();
  turns.begin();
  let calls = 0;
  let resolve!: () => void;
  const first = turns.stop(() => { calls += 1; return new Promise<void>((done) => { resolve = done; }); });
  const second = turns.stop(async () => { calls += 1; });
  await Promise.resolve();
  resolve();
  await Promise.all([first, second]);
  assert.equal(calls, 1);
  assert.equal(turns.busy, false);
});

test("a failed server stop still releases the turn", async () => {
  const turns = new ChatTurnController();
  turns.begin();
  await assert.rejects(turns.stop(async () => { throw new Error("offline"); }), /offline/);
  assert.equal(turns.busy, false);
  assert.equal(turns.isCurrent(turns.begin()), true);
});

test("a synchronously throwing stop callback does not poison later turns", async () => {
  const turns = new ChatTurnController();
  turns.begin();
  await assert.rejects(turns.stop(() => { throw new Error("invalid request"); }), /invalid request/);
  const next = turns.begin();
  await turns.stop(async () => undefined);
  assert.equal(next.controller.signal.aborted, true);
  assert.equal(turns.busy, false);
});
